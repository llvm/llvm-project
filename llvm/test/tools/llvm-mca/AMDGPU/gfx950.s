# RUN: llvm-mca -mtriple=amdgcn -mcpu=gfx950 --timeline --iterations=1 --timeline-max-cycles=0 < %s | FileCheck %s

# CHECK: Iterations:        1
# CHECK: Instructions:      15
# CHECK: Total Cycles:      60
# CHECK: Total uOps:        15

v_mfma_ld_scale_b32 v0, v0
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3]
v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15]

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


# CHECK:     [0]    [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT: -      -      -      -     1.00    -      -    v_mfma_ld_scale_b32 v0, v0
# CHECK-NEXT: -      -      -      -     1.00    -      -    v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[4:11], v[0:3]
# CHECK-NEXT: -      -      -      -     1.00    -      -    v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[4:11], v[4:11], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x32_f16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], v[0:3], a[4:7]
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x16_f16 v[0:15], v[0:3], v[0:3], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x16_f16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x16_bf16 v[0:15], v[0:3], v[0:3], v[0:15]
# CHECK-NEXT: -      -      -      -      -      -     8.00  v_mfma_f32_32x32x16_bf16 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2
# CHECK-NEXT: -      -      -      -     1.00    -      -    v_mfma_i32_16x16x64_i8 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
# CHECK-NEXT: -      -      -      -     1.00    -      -    v_mfma_i32_16x16x64_i8 a[0:3], v[0:3], v[0:3], a[4:7]
# CHECK-NEXT: -      -      -      -     1.00    -      -    v_mfma_i32_32x32x32_i8 v[0:15], v[0:3], v[0:3], v[0:15]
# CHECK-NEXT: -      -      -      -     1.00    -      -    v_mfma_i32_32x32x32_i8 a[0:15], a[0:3], a[0:3], a[0:15] blgp:2
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x32_bf16 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
# CHECK-NEXT: -      -      -      -      -      -     4.00  v_mfma_f32_16x16x32_bf16 a[0:3], v[0:3], v[0:3], a[4:7]
