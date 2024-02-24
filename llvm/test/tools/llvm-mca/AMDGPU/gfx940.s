# RUN: llvm-mca -mtriple=amdgcn -mcpu=gfx940 --timeline --iterations=1 --timeline-max-cycles=0 < %s | FileCheck %s

# CHECK: Iterations:        1
# CHECK: Instructions:      21
# CHECK: Total Cycles:      102
# CHECK: Total uOps:        27

v_pk_fma_f32 v[0:1], v[0:1], v[0:1], v[0:1]
v_pk_mov_b32 v[0:1], v[2:3], v[4:5]
v_pk_add_f32 v[0:1], v[0:1], v[0:1]
v_pk_mul_f32 v[0:1], v[0:1], v[0:1]
v_add_co_u32 v5, s[0:1], v1, v2
v_sub_co_u32 v5, s[0:1], v1, v2
v_subrev_co_u32 v5, s[0:1], v1, v2
v_addc_co_u32 v5, s[0:1], v1, v2, s[2:3]
v_subb_co_u32 v5, s[0:1], v1, v2, s[2:3]
v_subbrev_co_u32 v5, s[0:1], v1, v2, s[2:3]
v_add_u32 v5, v1, v2
v_sub_u32 v5, v1, v2
v_subrev_u32 v5, v1, v2

v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5]
v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5]

v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33]
v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33]

v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3]
v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], v[2:3], v[2:3]

v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7]
v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7]

# CHECK:     [0]    [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_pk_fma_f32 v[0:1], v[0:1], v[0:1], v[0:1]
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_pk_mov_b32 v[0:1], v[2:3], v[4:5]
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_pk_add_f32 v[0:1], v[0:1], v[0:1]
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_pk_mul_f32 v[0:1], v[0:1], v[0:1]
# CHECK-NEXT: -      -      -     1.00   1.00    -      -     v_add_co_u32_e64 v5, s[0:1], v1, v2
# CHECK-NEXT: -      -      -     1.00   1.00    -      -     v_sub_co_u32_e64 v5, s[0:1], v1, v2
# CHECK-NEXT: -      -      -     1.00   1.00    -      -     v_subrev_co_u32_e64 v5, s[0:1], v1, v2
# CHECK-NEXT: -      -      -     1.00   1.00    -      -     v_addc_co_u32_e64 v5, s[0:1], v1, v2, s[2:3]
# CHECK-NEXT: -      -      -     1.00   1.00    -      -     v_subb_co_u32_e64 v5, s[0:1], v1, v2, s[2:3]
# CHECK-NEXT: -      -      -     1.00   1.00    -      -     v_subbrev_co_u32_e64 v5, s[0:1], v1, v2, s[2:3]
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_add_u32_e32 v5, v1, v2
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_sub_u32_e32 v5, v1, v2
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_subrev_u32_e32 v5, v1, v2
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[2:5]
# CHECK-NEXT: -      -      -      -      -      -     8.00   v_mfma_f32_16x16x4_f32 v[0:3], v0, v1, v[2:5]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[18:33]
# CHECK-NEXT: -      -      -      -      -      -     16.00  v_mfma_f32_32x32x2_f32 v[0:15], v0, v1, v[18:33]
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3]
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], v[2:3], v[2:3]
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7]
# CHECK-NEXT: -      -      -      -     1.00    -      -     v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7]
