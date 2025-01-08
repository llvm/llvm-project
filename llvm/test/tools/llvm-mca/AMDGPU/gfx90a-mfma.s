# RUN: llvm-mca -mtriple=amdgcn -mcpu=gfx90a --timeline --iterations=1 --timeline-max-cycles=0 < %s | FileCheck %s

# CHECK: Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects (U)

# CHECK:     [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT: 1      8     4.00                  U     v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], a[0:1]
# CHECK-NEXT: 1      8     4.00                  U     v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], v[0:1]
# CHECK-NEXT: 1      12    8.00                  U     v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], a[0:7]
# CHECK-NEXT: 1      12    8.00                  U     v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[0:7]


# CHECK: Resources:
# CHECK-NEXT: [0]   - HWBranch
# CHECK-NEXT: [1]   - HWExport
# CHECK-NEXT: [2]   - HWLGKM
# CHECK-NEXT: [3]   - HWSALU
# CHECK-NEXT: [4]   - HWVALU
# CHECK-NEXT: [5]   - HWVMEM
# CHECK-NEXT: [6]   - HWXDL

# CHECK:     [0]    [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT: -      -      -      -     4.00    -      -     v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], a[0:1]
# CHECK-NEXT: -      -      -      -     4.00    -      -     v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], v[0:1]
# CHECK-NEXT: -      -      -      -     8.00    -      -     v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], a[0:7]
# CHECK-NEXT: -      -      -      -     8.00    -      -     v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[0:7]
v_mfma_f64_4x4x4f64 a[0:1], v[0:1], v[2:3], a[0:1]
v_mfma_f64_4x4x4f64 v[0:1], v[0:1], v[2:3], v[0:1]


v_mfma_f64_16x16x4f64 a[0:7], v[0:1], v[2:3], a[0:7]
v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[0:7]

