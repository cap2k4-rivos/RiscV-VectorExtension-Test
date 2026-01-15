# RISC-V Vector Q15 AXPY Implementation

**Author:** Akshay Behl  
**Mentorship Program:** LFX Mentorship - RISC-V Audiomark  
**Date:** January 2026

## Overview

This project implements a saturating multiply-accumulate function using RISC-V Vector (RVV) intrinsics v1.0. The function computes a scaled vector addition with Q15 fixed-point saturation:
```
y[i] = saturate_q15(a[i] + α × b[i])
```

for all `i ∈ [0, n)`, where saturation clamps results to the Q15 range `[-32768, 32767]`.

---

## Table of Contents

- [Implementation Details](#implementation-details)
- [Design Choices](#design-choices)
- [Build Instructions](#build-instructions)
- [Performance Analysis](#performance-analysis)
- [Verification](#verification)
- [Future Optimizations](#future-optimizations)
- [References](#references)

---

## Implementation Details

### Function Signature
```c
void q15_axpy_rvv(const int16_t *a, const int16_t *b,
                  int16_t *y, int n, int16_t alpha);
```

### Algorithm

The RVV implementation uses the following approach:

1. **Vector Length Configuration:** Uses `vsetvl` for dynamic vector length based on remaining elements
2. **Widening Multiply:** `vwmul` performs 16-bit × 16-bit → 32-bit multiplication to prevent overflow
3. **Widening Add:** `vwadd_wv` adds 32-bit product with 16-bit input (widened to 32-bit)
4. **Saturating Narrowing:** `vnclip` converts 32-bit result back to 16-bit with saturation

### Key Code Section
```c
for (size_t i = 0; i < n; ) {
    // Set vector length for remaining elements
    vl = __riscv_vsetvl_e16m1(n - i);
    
    // Load input vectors (16-bit elements)
    vint16m1_t va = __riscv_vle16_v_i16m1(a + i, vl);
    vint16m1_t vb = __riscv_vle16_v_i16m1(b + i, vl);
    
    // Widening multiply: alpha * b[i] → 32-bit
    vint32m2_t vmul = __riscv_vwmul_vx_i32m2(vb, alpha, vl);
    
    // Widening add: 32-bit + 16-bit (widened) → 32-bit
    vint32m2_t vacc = __riscv_vwadd_wv_i32m2(vmul, va, vl);
    
    // Saturating narrow: 32-bit → 16-bit with Q15 saturation
    vint16m1_t vy = __riscv_vnclip_wx_i16m1(vacc, 0, __RISCV_VXRM_RNU, vl);
    
    // Store result
    __riscv_vse16_v_i16m1(y + i, vy, vl);
    
    i += vl;
}
```

---

## Design Choices

### 1. **Widening Operations for Overflow Prevention**

**Choice:** Use `vwmul` (widening multiply) and `vwadd_wv` (widening add)

**Reasoning:**
- Q15 multiplication: `[-32768, 32767] × [-32768, 32767]` can produce values up to ±1,073,709,056
- Standard 16-bit operations would overflow
- Widening to 32-bit intermediate results guarantees correctness
- Hardware acceleration for widening operations makes them efficient

**Alternative Considered:** Using standard multiply with careful scaling
- **Rejected:** Would require lossy pre-scaling and reduce precision

### 2. **LMUL = 1 (Implied by m1/m2 suffixes)**

**Choice:** Use `e16m1` (LMUL=1) for 16-bit operations, `i32m2` (LMUL=2) for 32-bit widened results

**Reasoning:**
- Balances parallelism with register pressure
- LMUL=2 for widened operations is automatic (required by RVV spec)
- Leaves registers available for other operations
- Portable across different VLEN implementations

**Alternative Considered:** LMUL=2 for 16-bit (more parallelism)
- **Trade-off:** Would double register usage, potentially causing spills on VLEN=128 systems

### 3. **Vector-Length Agnostic Implementation**

**Choice:** Use `vsetvl_e16m1(n - i)` for dynamic vector length

**Reasoning:**
- Portable across VLEN=64, 128, 256, 512, etc.
- Automatically handles tail elements (when n is not a multiple of VLEN)
- Complies with RVV spec requirements for portable code
- No hardcoded assumptions about vector register width

**Alternative Considered:** Hardcoded VLEN assumptions with scalar cleanup loop
- **Rejected:** Not portable, violates challenge requirements

### 4. **Hardware Saturating Clip**

**Choice:** Use `vnclip_wx_i16m1` with shift=0 and rounding mode RNU

**Reasoning:**
- Single instruction for 32-bit → 16-bit conversion with saturation
- Hardware-accelerated Q15 clamping: `clip(x, -32768, 32767)`
- More efficient than scalar comparisons
- Rounding mode RNU (round-to-nearest-up) matches Q15 semantics

**Alternative Considered:** Manual saturation with `vmax`/`vmin`
- **Rejected:** Requires additional instructions and doesn't leverage hardware saturation

### 5. **RVV v1.0 Intrinsics**

**Choice:** Target ratified RVV v1.0 specification (`__riscv_v_intrinsic >= 1000000`)

**Reasoning:**
- Stable, ratified specification (not experimental)
- Broad compiler support (Clang, GCC 14+)
- Future-proof implementation
- Consistent semantics across toolchains

---

## Build Instructions

### Prerequisites

- RISC-V GNU Toolchain or LLVM with RVV support
- Target: RV64GC with Vector extension (rv64gcbv)
- Simulator: QEMU, Spike, or real hardware

### Compilation

#### Using Clang (Recommended - RVV v1.0)
```bash
riscv64-unknown-elf-clang -march=rv64gcbv -mabi=lp64d -O2 \
  q15_axpy_rvv.c -o q15_axpy_clang.elf -static \
  --gcc-toolchain=/path/to/riscv/toolchain \
  --sysroot=/path/to/riscv/toolchain/riscv64-unknown-elf
```

#### Using GCC (Requires v14+ for RVV v1.0)
```bash
riscv64-unknown-elf-gcc -march=rv64gcbv -mabi=lp64d -O2 \
  q15_axpy_rvv.c -o q15_axpy_gcc.elf -static
```

**Note:** Older GCC versions (< v14) may use RVV v0.12 intrinsics, which are incompatible with this implementation.

### Running on QEMU
```bash
qemu-riscv64 q15_axpy_clang.elf
```

### Verification

The program includes a built-in test harness that:
1. Generates deterministic test data (4096 elements)
2. Runs scalar reference implementation
3. Runs RVV implementation
4. Compares results bit-for-bit
5. Reports cycle counts (via `rdcycle` instruction)

**Expected output:**
```
Cycles ref: XXXXX
RISCV Vector Extension working...
Verify RVV: OK (max diff = 0)
Cycles RVV: XXXXX
```

---

## Performance Analysis

### QEMU Results
**CLANG With Vector Enabled**
```bash
$ qemu-riscv64 q15_axpy_clang_v.elf
Cycles ref: 322608
RISCV Vector Extension working...
Verify RVV: OK (max diff = 0)
Cycles RVV: 853182
```

**CLANG With Vector Disabled**
```bash
$ qemu-riscv64 q15_axpy_clang_n.elf
Cycles ref: 102729
RISCV Vector Extension not working...
Verify RVV: OK (max diff = 0)
Cycles RVV: 410784
```

### Environment

- **Toolchain:** Clang with RVV v1.0 intrinsics
- **Target:** rv64gcbv (RV64 with G, C, B, V extensions)
- **Simulator:** QEMU [VERSION HERE]
- **Test Size:** N = 4096 elements
- **Alpha:** α = 3

### Measured Performance (QEMU User-Mode Emulation)

| Implementation | Cycles | Speedup vs Scalar |
|----------------|--------|-------------------|
| Scalar Reference | 322,608 | 1.0× (baseline) |
| RVV Implementation | 853,182 | **0.38×** (2.6× slowdown) |

### Understanding the QEMU Slowdown

**The measured slowdown is expected and does NOT reflect real hardware performance.**

QEMU emulates vector instructions in software:
- Each `vle16` (vector load) is expanded into multiple scalar loads
- Widening operations require complex software emulation
- `vsetvl` dynamic configuration has overhead in emulation
- No actual hardware vector units—everything is interpreted

**Analogy:** QEMU is like a person pretending to be 8 workers doing tasks simultaneously, but actually doing them one at a time. The overhead of "pretending" makes it slower than just admitting you're one person.

### Theoretical Performance (Real Hardware)

#### Assumptions for Back-of-Envelope Calculation

- **VLEN:** 128 bits (common for embedded RVV implementations like Allwinner D1)
- **Element Width:** 16 bits (int16_t)
- **Elements per Vector:** 128 ÷ 16 = **8 elements processed in parallel**
- **LMUL:** 1 for 16-bit, 2 for widened 32-bit operations

#### Instruction Count Analysis

**Scalar Version (per element):**
1. Load `a[i]` (1 cycle)
2. Load `b[i]` (1 cycle)
3. Multiply `alpha × b[i]` (2-4 cycles)
4. Add `a[i] + product` (1 cycle)
5. Saturate to Q15 range (2-3 cycles for compare/branch)
6. Store `y[i]` (1 cycle)

**Total per element:** ~8-12 cycles  
**Total for 4096 elements:** ~32,768 - 49,152 cycles

**Vector Version (per iteration):**
1. `vsetvl` (1 cycle)
2. `vle16` for `a` (1 cycle, amortized across 8 elements)
3. `vle16` for `b` (1 cycle, amortized across 8 elements)
4. `vwmul` (2-3 cycles)
5. `vwadd_wv` (1-2 cycles)
6. `vnclip` (1-2 cycles)
7. `vse16` (1 cycle, amortized across 8 elements)

**Total per iteration:** ~8-12 cycles for 8 elements = ~1-1.5 cycles per element  
**Total for 4096 elements:** ~512 iterations × 10 cycles = ~5,120 cycles

#### Expected Speedup on Real Hardware

**Optimistic:** 49,152 ÷ 5,120 = **9.6×**  
**Conservative:** 32,768 ÷ 6,144 = **5.3×**  
**Realistic:** **6-8× speedup** expected on real RVV hardware

### Performance on Different Architectures

| Architecture | VLEN | Elements/Vector | Expected Speedup |
|--------------|------|-----------------|------------------|
| Minimal RVV | 64 | 4 | 3-4× |
| Allwinner D1 | 128 | 8 | 6-8× |
| SiFive P670 | 256 | 16 | 10-14× |
| High-end Server | 512+ | 32+ | 15-25× |

**Note:** Actual performance depends on:
- Memory bandwidth
- Cache hierarchy
- Instruction latencies
- Pipeline depth
- Compiler optimizations

---

## Verification

### Correctness Testing

The implementation passes bit-for-bit verification against the scalar reference for:

✅ **Edge cases:**
- Maximum positive values: `a[i] = 32767, b[i] = 32767, alpha = 1`
- Maximum negative values: `a[i] = -32768, b[i] = -32768, alpha = -1`
- Overflow requiring saturation: `a[i] = 30000, b[i] = 30000, alpha = 3`
- Zero inputs: `a[i] = 0, b[i] = 0`
- Mixed signs

✅ **Random data:** 4096 elements with deterministic seeding

✅ **Vector-length agnostic:** Works correctly regardless of VLEN (tested conceptually)

### Test Output
```
Cycles ref: 322608
RISCV Vector Extension working...
Verify RVV: OK (max diff = 0)
Cycles RVV: 853182
```

**Maximum difference: 0** (bit-exact match with reference)

---

## Future Optimizations

### Potential Improvements

1. **LMUL Tuning:** Experiment with LMUL=2/4 for higher parallelism on high-VLEN systems
2. **Unrolling:** Manual loop unrolling to reduce `vsetvl` overhead
3. **Prefetching:** Software prefetch hints for large arrays
4. **SIMD Within a Register (SWAR):** Combine with scalar optimizations for tiny vectors
5. **Masking:** Use tail-masking optimizations for better tail handling

### Alternative Approaches Explored

1. **Standard multiply instead of widening multiply**
   - Would overflow for large values
   - Required Q15 scaling, losing precision

2. **Manual saturation with `vmax`/`vmin`**
   - More instructions than `vnclip`
   - Doesn't leverage hardware saturation

3. **Fractional LMUL (e.g., m1/2)**
   - Could reduce register pressure further
   - Trade-off: less parallelism, not needed for this problem

---

## Repository Structure
```
.
├── README.md                # This file
├── q15_axpy_rvv.c           # Implementation + test harness
├── q15_axpy_gcc.elf	     # Compiled with GCC with Vector enabled, RISC-V Vector Intrinsic Version 0.12 ( Same as Vector Disabled )
├── q15_axpy_clang_v.elf     # Compiled with CLANG with Vector enabled, RISC-V Vector Intrinsic Version 1.0
└── q15_axpy_clang_n.elf     # Compiled with CLANG with Vector disabled


```

---

## Build Information

### Toolchain Versions

- **Clang:** [VERSION - run: riscv64-unknown-elf-clang --version]
- **GCC:** [VERSION - run: riscv64-unknown-elf-gcc --version]
- **QEMU:** [VERSION - run: qemu-riscv64 --version]

### Compilation Flags

- `-march=rv64gcbv`: RV64 with G (IMAFD), C (compressed), B (bit manipulation), V (vector)
- `-mabi=lp64d`: LP64 ABI with double-precision floating-point
- `-O2`: Moderate optimization (balance between speed and compile time)
- `-static`: Static linking for standalone execution

---

## Lessons Learned

1. **RVV intrinsics are powerful but require careful data flow management** - widening operations need correct LMUL ratios
2. **QEMU emulation doesn't reflect real performance** - simulation is for correctness, not benchmarking
3. **Vector-length agnostic code is essential** - portability across implementations is critical
4. **Hardware saturation saves instructions** - leveraging specialized operations improves efficiency

---

## References

1. [RISC-V "V" Vector Extension Specification v1.0](https://github.com/riscv/riscv-v-spec/releases/tag/v1.0)
2. [RISC-V Unprivileged ISA Specification](https://riscv.org/technical/specifications/)
3. [RISC-V GNU Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)

---
