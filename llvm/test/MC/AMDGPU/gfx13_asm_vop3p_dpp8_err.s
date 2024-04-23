// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 %s 2>&1 | FileCheck --check-prefix=GFX13 --implicit-check-not=error: %s

v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 neg_lo:[0,1,1] neg_hi:[1,0,1] dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 dpp8:[0,1,2,3,4,5,6,7]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 dpp8:[0,1,2,3,4,5,6,7]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 dpp8:[0,1,2,3,4,5,6,7]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
