# RUN: llvm-mca -mtriple=amdgcn -mcpu=gfx942 --timeline --iterations=1 --timeline-max-cycles=0 < %s | FileCheck %s

# CHECK: Iterations:        1
# CHECK: Instructions:      78
# CHECK: Total Cycles:      701
# CHECK: Total uOps:        78

v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5]
v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5]

v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33]
v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33]

v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3]
v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], v[2:3], v[2:3]

v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7]
v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7]

v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
v_mfma_f32_16x16x16_f16 a[0:3], v[4:5], v[6:7], a[0:3]

v_mfma_f32_32x32x8_f16 v[0:15], v[4:5], v[6:7], v[0:15]
v_mfma_f32_32x32x8_f16 a[0:15], v[4:5], v[6:7], a[0:15]

v_mfma_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3]
v_mfma_f32_16x16x16_bf16 a[0:3], v[4:5], v[6:7], a[0:3]

v_mfma_f32_32x32x8_bf16 v[0:15], v[4:5], v[6:7], v[0:15]
v_mfma_f32_32x32x8_bf16 a[0:15], v[4:5], v[6:7], a[0:15]

v_mfma_i32_16x16x32_i8 v[0:3], v[4:5], v[6:7], v[0:3]
v_mfma_i32_16x16x32_i8 a[0:3], v[4:5], v[6:7], a[0:3]

v_mfma_i32_32x32x16_i8 v[0:15], v[2:3], v[4:5], v[0:15]
v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15]

v_mfma_f32_4x4x4_16b_f16 v[0:3], v[0:1], v[2:3], v[2:5]
v_mfma_f32_4x4x4_16b_f16 a[0:3], v[0:1], v[2:3], a[2:5]

v_mfma_f32_16x16x4_4b_f16 v[0:15], v[2:3], v[4:5], v[18:33]
v_mfma_f32_16x16x4_4b_f16 a[0:15], v[2:3], v[4:5], a[18:33]

v_mfma_f32_32x32x4_2b_f16 v[0:31], v[0:1], v[2:3], v[34:65]
v_mfma_f32_32x32x4_2b_f16 a[0:31], v[0:1], v[2:3], a[34:65]

v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[0:1], v[2:3], v[2:5]
v_mfma_f32_4x4x4_16b_bf16 a[0:3], v[0:1], v[2:3], a[2:5]

v_mfma_f32_16x16x4_4b_bf16 v[0:15], v[2:3], v[4:5], v[18:33]
v_mfma_f32_16x16x4_4b_bf16 a[0:15], v[2:3], v[4:5], a[18:33]

v_mfma_f32_32x32x4_2b_bf16 v[0:31], v[0:1], v[2:3], v[34:65]
v_mfma_f32_32x32x4_2b_bf16 a[0:31], v[0:1], v[2:3], a[34:65]

v_mfma_f32_4x4x1_16b_f32 v[0:3], v0, v1, v[2:5]
v_mfma_f32_4x4x1_16b_f32 a[0:3], v0, v1, a[2:5]

v_mfma_f32_16x16x1_4b_f32 v[0:15], v0, v1, v[18:33]
v_mfma_f32_16x16x1_4b_f32 a[0:15], v0, v1, a[18:33]

v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5]
v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5]

v_mfma_f32_32x32x1_2b_f32 v[0:31], v0, v1, v[34:65] blgp:7
v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[34:65] blgp:7

v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33]
v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33]

v_mfma_i32_4x4x4_16b_i8 v[0:3], v0, v1, v[2:5]
v_mfma_i32_4x4x4_16b_i8 a[0:3], v0, v1, a[2:5]

v_mfma_i32_16x16x4_4b_i8 v[0:15], v0, v1, v[18:33]
v_mfma_i32_16x16x4_4b_i8 a[0:15], v0, v1, a[18:33]

v_mfma_i32_32x32x4_2b_i8 v[0:31], v0, v1, v[34:65]
v_mfma_i32_32x32x4_2b_i8 a[0:31], v0, v1, a[34:65]

v_smfmac_f32_16x16x32_f16 v[10:13], a[2:3], v[4:7], v0 cbsz:3 abid:1
v_smfmac_f32_16x16x32_f16 a[10:13], v[2:3], a[4:7], v1

v_smfmac_f32_32x32x16_f16 v[10:25], a[2:3], v[4:7], v2 cbsz:3 abid:1
v_smfmac_f32_32x32x16_f16 a[10:25], v[2:3], a[4:7], v3

v_smfmac_f32_16x16x32_bf16 v[10:13], a[2:3], v[4:7], v4 cbsz:3 abid:1
v_smfmac_f32_16x16x32_bf16 a[10:13], v[2:3], a[4:7], v5

v_smfmac_i32_16x16x64_i8 v[10:13], a[2:3], v[4:7], v8 cbsz:3 abid:1
v_smfmac_i32_16x16x64_i8 a[10:13], v[2:3], a[4:7], v9

v_smfmac_i32_32x32x32_i8 v[10:25], a[2:3], v[4:7], v10 cbsz:3 abid:1
v_smfmac_i32_32x32x32_i8 a[10:25], v[2:3], a[4:7], v11

v_mfma_f32_16x16x32_bf8_bf8 v[0:3], v[2:3], v[4:5], v[0:3]
v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]

v_mfma_f32_16x16x32_bf8_fp8 v[0:3], v[2:3], v[4:5], v[0:3]
v_mfma_f32_16x16x32_bf8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]

v_mfma_f32_16x16x32_fp8_bf8 v[0:3], v[2:3], v[4:5], v[0:3]
v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]

v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[2:3], v[4:5], v[0:3]
v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]

v_mfma_f32_32x32x16_bf8_bf8 v[0:15], v[2:3], v[4:5], v[0:15]
v_mfma_f32_32x32x16_fp8_bf8 v[0:15], v[2:3], v[4:5], v[0:15]
v_mfma_f32_32x32x16_bf8_fp8 v[0:15], v[2:3], v[4:5], v[0:15]
v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[2:3], v[4:5], v[0:15]

v_smfmac_f32_16x16x64_bf8_bf8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
v_smfmac_f32_16x16x64_bf8_fp8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
v_smfmac_f32_16x16x64_fp8_bf8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
v_smfmac_f32_16x16x64_fp8_fp8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1

v_smfmac_f32_32x32x32_bf8_bf8 v[0:15], v[2:3], v[4:7], v1 cbsz:3 abid:1
v_smfmac_f32_32x32x32_bf8_fp8 v[0:15], v[2:3], v[4:7], v1 cbsz:3 abid:1
v_smfmac_f32_32x32x32_fp8_bf8 v[0:15], v[2:3], v[4:7], v1 cbsz:3 abid:1
v_smfmac_f32_32x32x32_fp8_fp8 v[0:15], v[2:3], v[4:7], v1 cbsz:3 abid:1

# CHECK: Instruction Info:
# CHECK-NEXT:[1]: #uOps
# CHECK-NEXT:[2]: Latency
# CHECK-NEXT:[3]: RThroughput
# CHECK-NEXT:[4]: MayLoad
# CHECK-NEXT:[5]: MayStore
# CHECK-NEXT:[6]: HasSideEffects (U)

# CHECK:     [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK:      1      8     4.00                  U     v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3]
# CHECK-NEXT: 1      8     4.00                  U     v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], v[2:3], v[2:3]
# CHECK-NEXT: 1      12    8.00                  U     v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7]
# CHECK-NEXT: 1      12    8.00                  U     v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7]

# CHECK: Resources:
# CHECK: [0]   - HWBranch
# CHECK: [1]   - HWExport
# CHECK: [2]   - HWLGKM
# CHECK: [3]   - HWSALU
# CHECK: [4]   - HWVALU
# CHECK: [5]   - HWVMEM
# CHECK: [6]   - HWXDL

# CHECK:     [0]    [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33]
# CHECK-NEXT: -      -      -      -     4.00    -      -     v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3]
# CHECK-NEXT: -      -      -      -     4.00    -      -     v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], v[2:3], v[2:3]
# CHECK-NEXT: -      -      -      -     8.00    -      -     v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7]
# CHECK-NEXT: -      -      -      -     8.00    -      -     v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x16_f16 a[0:3], v[4:5], v[6:7], a[0:3]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_32x32x8_f16 v[0:15], v[4:5], v[6:7], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_32x32x8_f16 a[0:15], v[4:5], v[6:7], a[0:15]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x16_bf16 a[0:3], v[4:5], v[6:7], a[0:3]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_32x32x8_bf16 v[0:15], v[4:5], v[6:7], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_32x32x8_bf16 a[0:15], v[4:5], v[6:7], a[0:15]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_i32_16x16x32_i8 v[0:3], v[4:5], v[6:7], v[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_i32_16x16x32_i8 a[0:3], v[4:5], v[6:7], a[0:3]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_i32_32x32x16_i8 v[0:15], v[2:3], v[4:5], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15]
# CHECK-NEXT: -      -      -      -      -      -     2.00   v_mfma_f32_4x4x4_16b_f16 v[0:3], v[0:1], v[2:3], v[2:5]
# CHECK-NEXT: -      -      -      -      -      -     2.00   v_mfma_f32_4x4x4_16b_f16 a[0:3], v[0:1], v[2:3], a[2:5]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_4b_f16 v[0:15], v[2:3], v[4:5], v[18:33]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_4b_f16 a[0:15], v[2:3], v[4:5], a[18:33]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x4_2b_f16 v[0:31], v[0:1], v[2:3], v[34:65]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x4_2b_f16 a[0:31], v[0:1], v[2:3], a[34:65]
# CHECK-NEXT: -      -      -      -      -      -     2.00   v_mfma_f32_4x4x4_16b_bf16 v[0:3], v[0:1], v[2:3], v[2:5]
# CHECK-NEXT: -      -      -      -      -      -     2.00   v_mfma_f32_4x4x4_16b_bf16 a[0:3], v[0:1], v[2:3], a[2:5]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_4b_bf16 v[0:15], v[2:3], v[4:5], v[18:33]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_4b_bf16 a[0:15], v[2:3], v[4:5], a[18:33]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x4_2b_bf16 v[0:31], v[0:1], v[2:3], v[34:65]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x4_2b_bf16 a[0:31], v[0:1], v[2:3], a[34:65]
# CHECK-NEXT: -      -      -      -      -      -     2.00   v_mfma_f32_4x4x1_16b_f32 v[0:3], v0, v1, v[2:5]
# CHECK-NEXT: -      -      -      -      -      -     2.00   v_mfma_f32_4x4x1_16b_f32 a[0:3], v0, v1, a[2:5]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x1_4b_f32 v[0:15], v0, v1, v[18:33]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x1_4b_f32 a[0:15], v0, v1, a[18:33]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x1_2b_f32 v[0:31], v0, v1, v[34:65] blgp:7
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[34:65] blgp:7
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33]
# CHECK-NEXT: -      -      -      -      -      -     2.00   v_mfma_i32_4x4x4_16b_i8 v[0:3], v0, v1, v[2:5]
# CHECK-NEXT: -      -      -      -      -      -     2.00   v_mfma_i32_4x4x4_16b_i8 a[0:3], v0, v1, a[2:5]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_i32_16x16x4_4b_i8 v[0:15], v0, v1, v[18:33]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_i32_16x16x4_4b_i8 a[0:15], v0, v1, a[18:33]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_i32_32x32x4_2b_i8 v[0:31], v0, v1, v[34:65]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_i32_32x32x4_2b_i8 a[0:31], v0, v1, a[34:65]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x32_f16 v[10:13], a[2:3], v[4:7], v0 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x32_f16 a[10:13], v[2:3], a[4:7], v1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x16_f16 v[10:25], a[2:3], v[4:7], v2 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x16_f16 a[10:25], v[2:3], a[4:7], v3
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x32_bf16 v[10:13], a[2:3], v[4:7], v4 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x32_bf16 a[10:13], v[2:3], a[4:7], v5
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_i32_16x16x64_i8 v[10:13], a[2:3], v[4:7], v8 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_i32_16x16x64_i8 a[10:13], v[2:3], a[4:7], v9
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_i32_32x32x32_i8 v[10:25], a[2:3], v[4:7], v10 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_i32_32x32x32_i8 a[10:25], v[2:3], a[4:7], v11
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x32_bf8_bf8 v[0:3], v[2:3], v[4:5], v[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x32_bf8_fp8 v[0:3], v[2:3], v[4:5], v[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x32_bf8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x32_fp8_bf8 v[0:3], v[2:3], v[4:5], v[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[2:3], v[4:5], v[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_32x32x16_bf8_bf8 v[0:15], v[2:3], v[4:5], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_32x32x16_fp8_bf8 v[0:15], v[2:3], v[4:5], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_32x32x16_bf8_fp8 v[0:15], v[2:3], v[4:5], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[2:3], v[4:5], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x64_bf8_bf8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x64_bf8_fp8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x64_fp8_bf8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x64_fp8_fp8 v[0:3], a[2:3], v[4:7], v1 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x32_bf8_bf8 v[0:15], v[2:3], v[4:7], v1 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x32_bf8_fp8 v[0:15], v[2:3], v[4:7], v1 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x32_fp8_bf8 v[0:15], v[2:3], v[4:7], v1 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x32_fp8_fp8 v[0:15], v[2:3], v[4:7], v1 cbsz:3 abid:1
