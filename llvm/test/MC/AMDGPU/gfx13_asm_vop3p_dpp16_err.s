// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 %s 2>&1 | FileCheck --check-prefix=GFX13 --implicit-check-not=error: %s

v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[0,0,0] neg_hi:[0,0,0] quad_perm:[2,2,3,1] bound_ctrl:0 fi:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 row_share:15
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 row_shl:15
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 row_mirror
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
