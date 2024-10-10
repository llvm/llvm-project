// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding < %s 2>&1 | FileCheck -strict-whitespace -implicit-check-not=error: -check-prefix=GFX13-ERR %s

s_waitcnt 0
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_wait_bvhcnt 0
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

// For v_dual_cndmask_b32 use of the explicit src2 forces VOPD3 form even if it is vcc_lo.
// If src2 is omitted then it forces VOPD form. As a result a proper form of the instruction
// has to be used if the other component of the dual instruction cannot be used if that
// encoding.

v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid VOPDY instruction
// GFX13-ERR: v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX13-ERR:                                  ^

v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX13-ERR: v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX13-ERR: ^

v_dual_cndmask_b32 v7, v1, v2 :: v_dual_cndmask_b32 v2, v4, v1, vcc_lo
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR: v_dual_cndmask_b32 v7, v1, v2 :: v_dual_cndmask_b32 v2, v4, v1, vcc_lo
// GFX13-ERR:                                                                 ^

v_dual_cndmask_b32 v7, v1, v2, vcc_lo :: v_dual_cndmask_b32 v2, v4, v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX13-ERR: v_dual_cndmask_b32 v7, v1, v2, vcc_lo :: v_dual_cndmask_b32 v2, v4, v1
// GFX13-ERR: ^

v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid VOPDY instruction
// GFX13-ERR: v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX13-ERR:                                  ^

v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX13-ERR: v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX13-ERR: ^

s_barrier_init 0
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

rts_trace_ray [v0, v[1:3], v[4], v[5:7], v[8:10]], s[4:7]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must set modifier r128=1

rts_trace_ray_nonblock v14, [v0, v[1:3], v[4], v[5:7], v[8:10]], s[4:7]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must set modifier r128=1

rts_read_vertex v[0:8], [v9, v10, v11], null
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must set modifier r128=1
