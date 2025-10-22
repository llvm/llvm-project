# RUN: llvm-mca -mtriple=amdgcn -mcpu=gfx950 --timeline --iterations=1 --timeline-max-cycles=0 < %s | FileCheck %s

# CHECK: Iterations:        1
# CHECK: Instructions:      133
# CHECK: Total Cycles:      1101
# CHECK: Total uOps:        133

v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], v[0:3], a[4:7]
v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15]
v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2
v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15]
v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2
v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
v_mfma_i32_16x16x64_i8 a[0:3], v[0:3], v[0:3], a[4:7]
v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15]
v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2
v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[0:3], a[4:7]

v_mfma_ld_scale_b32 v0, v0

;; FIXME: should have different cycle count depending on whether either matrix is f8
;;  TODO: test vdc/adc
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] blgp:1
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:9], v[0:3] blgp:2
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:9], v[0:3] blgp:3
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:7], v[0:3] blgp:4
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] cbsz:1
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:2
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:3
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[4:11], v[0:3] cbsz:4
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:2 blgp:1
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:9], v[0:3] cbsz:1 blgp:2

;; FIXME: should have different cycle count depending on whether either matrix is f8
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] blgp:1
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:9], v[0:15] blgp:2
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:9], v[0:15] blgp:3
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:7], v[0:15] blgp:4
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] cbsz:1
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:2
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:3
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:7], v[4:11], v[0:15] cbsz:4
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:2
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] blgp:1

;; FIXME: should have different cycle count depending on whether either matrix is f8
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3], v5, v5
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3], v5, v5 blgp:1
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:9], v[0:3], v5, v5 cbsz:2 blgp:2

v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15], v5, v5
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15], v5, v5 cbsz:2 blgp:1
v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:9], v[0:15], v5, v5 cbsz:2 blgp:2

;;  TODO: These results are wrong
v_smfmac_f32_16x16x64_f16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_f32_32x32x32_f16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_f32_16x16x64_bf16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_f32_32x32x32_bf16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_i32_16x16x128_i8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_i32_32x32x64_i8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

v_smfmac_f32_16x16x128_bf8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_f32_16x16x128_bf8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_f32_16x16x128_fp8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_f32_16x16x128_fp8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1

v_smfmac_f32_32x32x64_bf8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_f32_32x32x64_bf8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_f32_32x32x64_fp8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
v_smfmac_f32_32x32x64_fp8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1

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

# CHECK:     [0]    [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], v[0:3], a[4:7]
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_i32_16x16x64_i8 a[0:3], v[0:3], v[0:3], a[4:7]
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[0:3], a[4:7]
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_mfma_ld_scale_b32 v0, v0
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3]
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] blgp:1
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:9], v[0:3] blgp:2
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:9], v[0:3] blgp:3
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:7], v[0:3] blgp:4
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3] cbsz:1
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:2
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:3
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[4:11], v[0:3] cbsz:4
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:11], v[0:3] cbsz:2 blgp:1
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:9], v[0:3] cbsz:1 blgp:2
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] blgp:1
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:9], v[0:15] blgp:2
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:9], v[0:15] blgp:3
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:7], v[0:15] blgp:4
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] cbsz:1
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:2
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:3
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:7], v[4:11], v[0:15] cbsz:4
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15] cbsz:2
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15] blgp:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3], v5, v5 op_sel_hi:[0,0,0]
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3], v5, v5 op_sel_hi:[0,0,0] blgp:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[4:9], v[0:3], v5, v5 op_sel_hi:[0,0,0] cbsz:2 blgp:2
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15], v5, v5 op_sel_hi:[0,0,0]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:11], v[0:15], v5, v5 op_sel_hi:[0,0,0] cbsz:2 blgp:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[4:9], v[4:9], v[0:15], v5, v5 op_sel_hi:[0,0,0] cbsz:2 blgp:2
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x64_f16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x32_f16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x64_bf16 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x32_bf16 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_i32_16x16x128_i8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_i32_32x32x64_i8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x128_bf8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x128_bf8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x128_fp8_bf8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     4.00   v_smfmac_f32_16x16x128_fp8_fp8 v[10:13], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x64_bf8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x64_bf8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x64_fp8_bf8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_smfmac_f32_32x32x64_fp8_fp8 v[10:25], a[2:5], v[4:11], v3 cbsz:3 abid:1
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33]
# CHECK-NEXT: -      -      -      -     4.00    -      -     v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3]
# CHECK-NEXT: -      -      -      -     4.00    -      -     v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], v[2:3], v[2:3]
# CHECK-NEXT: -      -      -      -     16.00   -      -     v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7]
# CHECK-NEXT: -      -      -      -     16.00   -      -     v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7]
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
