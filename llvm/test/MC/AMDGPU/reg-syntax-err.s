// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOVI %s

s_mov_b32 s1, s 1
// NOVI: error: invalid operand for instruction

s_mov_b32 s1, s[0 1
// NOVI: error: failed parsing operand

s_mov_b32 s1, s[0:0 1
// NOVI: error: failed parsing operand

s_mov_b32 s1, [s[0 1
// NOVI: error: failed parsing operand

s_mov_b32 s1, [s[0:1] 1
// NOVI: error: failed parsing operand

s_mov_b32 s1, [s0, 1
// NOVI: error: failed parsing operand

s_mov_b32 s1, s999 1
// NOVI: error: failed parsing operand

s_mov_b32 s1, s[1:2] 1
// NOVI: error: failed parsing operand

s_mov_b32 s1, s[0:2] 1
// NOVI: error: failed parsing operand

s_mov_b32 s1, xnack_mask_lo 1
// NOVI: error: failed parsing operand

s_mov_b32 s1, s s0
// NOVI: error: invalid operand for instruction

s_mov_b32 s1, s[0 s0
// NOVI: error: failed parsing operand

s_mov_b32 s1, s[0:0 s0
// NOVI: error: failed parsing operand

s_mov_b32 s1, [s[0 s0
// NOVI: error: failed parsing operand

s_mov_b32 s1, [s[0:1] s0
// NOVI: error: failed parsing operand

s_mov_b32 s1, [s0, s0
// NOVI: error: failed parsing operand

s_mov_b32 s1, s999 s0
// NOVI: error: failed parsing operand

s_mov_b32 s1, s[1:2] s0
// NOVI: error: failed parsing operand

s_mov_b32 s1, s[0:2] vcc_lo
// NOVI: error: failed parsing operand

s_mov_b32 s1, xnack_mask_lo s1
// NOVI: error: failed parsing operand

exp mrt0 v1, v2, v3, v4000 off
// NOVI: error: failed parsing operand

v_add_f64 v[0:1], v[0:1], v[0xF00000001:0x2]
// NOVI: error: failed parsing operand

v_add_f64 v[0:1], v[0:1], v[0x1:0xF00000002]
// NOVI: error: failed parsing operand

s_mov_b32 s1, s[0:-1]
// NOVI: error: failed parsing operand
