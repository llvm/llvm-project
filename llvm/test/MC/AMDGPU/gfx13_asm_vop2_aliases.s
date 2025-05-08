// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize32 -show-encoding %s | FileCheck --check-prefixes=GFX13 %s

v_min_f32 v5, v1, v2
// GFX13: v_min_num_f32_e32 v5, v1, v2            ; encoding: [0x01,0x05,0x0a,0x1e]

v_max_f32 v5, v1, v2
// GFX13: v_max_num_f32_e32 v5, v1, v2            ; encoding: [0x01,0x05,0x0a,0x20]

v_min_f16 v5, v1, v2
// GFX13: v_min_num_f16_e32 v5, v1, v2            ; encoding: [0x01,0x05,0x0a,0x74]

v_max_f16 v5, v1, v2
// GFX13: v_max_num_f16_e32 v5, v1, v2            ; encoding: [0x01,0x05,0x0a,0x72]

v_max_f64 v[5:6], v[1:2], v[2:3]
// GFX13: v_max_num_f64_e32 v[5:6], v[1:2], v[2:3] ; encoding: [0x01,0x05,0x0a,0x1c]

v_min_f64 v[5:6], v[1:2], v[2:3]
// GFX13: v_min_num_f64_e32 v[5:6], v[1:2], v[2:3] ; encoding: [0x01,0x05,0x0a,0x1a]
