// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOVI %s

s_mov_b32 s1, s 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s1, s[0 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s[0:0 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, [s[0 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, [s[0:1] 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, [s0, 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s999 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s[1:2] 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s[0:2] 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s1, xnack_mask_lo 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s1, s[0 s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s[0:0 s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, [s[0 s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, [s[0:1] s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, [s0, s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s999 s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s[1:2] s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s[0:2] vcc_lo
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s1, xnack_mask_lo s1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

exp mrt0 v1, v2, v3, v4000 off
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

v_add_f64 v[0:1], v[0:1], v[0xF00000001:0x2]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

v_add_f64 v[0:1], v[0:1], v[0x1:0xF00000002]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand

s_mov_b32 s1, s[0:-1]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand
