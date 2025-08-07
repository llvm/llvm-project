// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 %s 2>&1 | FileCheck %s -check-prefix=GFX13 --implicit-check-not=error: --strict-whitespace

v_dual_fma_f64 v[252:253], v[6:7], v[4:5], v[10:11] :: v_dual_add_f32 v8, v1, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_add_f64 v[252:253], v[6:7], v[4:5] :: v_dual_add_f32 v8, v1, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_mul_f64 v[252:253], v[6:7], v[4:5] :: v_dual_add_f32 v8, v1, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_num_f64 v[252:253], v[6:7], v[4:5] :: v_dual_add_f32 v8, v1, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_num_f64 v[252:253], v[6:7], v[4:5] :: v_dual_add_f32 v8, v1, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
