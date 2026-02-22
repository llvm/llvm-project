// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1150 -mattr=+wavefrontsize32 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1150 -mattr=+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_cvt_f32_bf8 v1, 3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8_sdwa v5, v1 src0_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8 v1, 3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8_sdwa v5, v1 src0_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf8_f32 v1, -v2, |v3|
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_bf8 v[0:1], v3 quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_bf8_dpp v[10:11], v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_bf8_sdwa v[10:11], v1 src0_sel:WORD_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_fp8 v[0:1], v3 quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_fp8_dpp v[10:11], v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_fp8_sdwa v[10:11], v1 src0_sel:WORD_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_fp8_f32 v1, -v2, |v3|
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_bf8_f32 v1, -|s2|, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_fp8_f32 v1, -|s2|, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
