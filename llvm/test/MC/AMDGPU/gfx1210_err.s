// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX1210-ERR --implicit-check-not=error: -strict-whitespace %s

s_set_vgpr_msb -1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: s_set_vgpr_msb accepts values in range [0..15]
// GFX1210-ERR: s_set_vgpr_msb -1
// GFX1210-ERR:                ^

s_set_vgpr_msb 16
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: s_set_vgpr_msb accepts values in range [0..15]
// GFX1210-ERR: s_set_vgpr_msb 16
// GFX1210-ERR:                ^

s_load_b32 s4, s[2:3], 10 th:TH_LOAD_NT th:TH_LOAD_NT
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_load_b32 s4, s[2:3], 10 th:TH_LOAD_NT th:TH_LOAD_NT
// GFX1210-ERR:                                         ^

s_load_b32 s4, s[2:3], 10 scope:SCOPE_SE scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_load_b32 s4, s[2:3], 10 scope:SCOPE_SE scope:SCOPE_SE
// GFX1210-ERR:                                          ^

s_load_b32 s4, s[2:3], 10 nv nv
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_load_b32 s4, s[2:3], 10 nv nv
// GFX1210-ERR:                              ^

v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR: v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1210-ERR:                          ^

// For v_dual_cndmask_b32 use of the explicit src2 forces VOPD3 form even if it is vcc_lo.
// If src2 is omitted then it forces VOPD form. As a result a proper form of the instruction
// has to be used if the other component of the dual instruction cannot be used if that
// encoding.

v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid VOPDY instruction
// GFX1210-ERR: v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX1210-ERR:                                  ^

v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX1210-ERR: v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX1210-ERR: ^

v_dual_cndmask_b32 v7, v1, v2 :: v_dual_cndmask_b32 v2, v4, v1, vcc_lo
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: v_dual_cndmask_b32 v7, v1, v2 :: v_dual_cndmask_b32 v2, v4, v1, vcc_lo
// GFX1210-ERR:                                                                 ^

v_dual_cndmask_b32 v7, v1, v2, vcc_lo :: v_dual_cndmask_b32 v2, v4, v1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX1210-ERR: v_dual_cndmask_b32 v7, v1, v2, vcc_lo :: v_dual_cndmask_b32 v2, v4, v1
// GFX1210-ERR: ^
