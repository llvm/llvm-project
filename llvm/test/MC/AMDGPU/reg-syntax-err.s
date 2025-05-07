// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOVI --implicit-check-not=error: %s

s_mov_b32 s1, s 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s1, s[0 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a closing square bracket

s_mov_b32 s1, s[0:0 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a closing square bracket

s_mov_b32 s1, [s[0 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a closing square bracket

s_mov_b32 s1, [s[0:1] 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a single 32-bit register

s_mov_b32 s1, [s0, 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a register or a list of registers

s_mov_b32 s1, s999 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_mov_b32 s1, s[1:2] 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment

s_mov_b32 s1, s[0:2] 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s1, xnack_mask_lo 1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: xnack_mask_lo register not available on this GPU

s_mov_b32 s1, s s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s1, s[0 s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a closing square bracket

s_mov_b32 s1, s[0:0 s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a closing square bracket

s_mov_b32 s1, [s[0 s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a closing square bracket

s_mov_b32 s1, [s[0:1] s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a single 32-bit register

s_mov_b32 s1, [s0, s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must have consecutive indices

s_mov_b32 s1, s999 s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_mov_b32 s1, s[1:2] s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment

s_mov_b32 s1, s[0:2] vcc_lo
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s1, xnack_mask_lo s1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: xnack_mask_lo register not available on this GPU

exp mrt0 v1, v2, v3, v4000 off
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

v_add_f64 v[0:1], v[0:1], v[0xF00000001:0x2]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register index

v_add_f64 v[0:1], v[0:1], v[0x1:0xF00000002]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register index

s_mov_b32 s1, s[0:-1]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register index

s_mov_b64 s[10:11], [exec_lo,vcc_hi]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register does not fit in the list

s_mov_b64 s[10:11], [exec_hi,exec_lo]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register does not fit in the list

s_mov_b64 s[10:11], [exec_lo,exec_lo]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register does not fit in the list

s_mov_b64 s[10:11], [exec,exec_lo]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register does not fit in the list

s_mov_b64 s[10:11], [exec_lo,exec]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register does not fit in the list

s_mov_b64 s[10:11], [exec_lo,s0]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s0,exec_lo]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s0,exec]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s0,v1]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [v0,s1]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s0,s0]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [s0,s2]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [s2,s1]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [a0,a2]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [a0,v1]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: missing register index

s_mov_b64 s[10:11], s[1:0]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: first register index should not exceed second index

s_mov_b64 s[10:11], [x0,s1]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register name

s_mov_b64 s[10:11], [s,s1]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: missing register index

s_mov_b64 s[10:11], [s01,s1]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [s0x]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register index

s_mov_b64 s[10:11], [s[0:1],s[2:3]]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a single 32-bit register

s_mov_b64 s[10:11], [s0,s[2:3]]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a single 32-bit register

s_mov_b64 s[10:11], [s0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a comma or a closing square bracket

s_mov_b64 s[10:11], [s0,s1
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected a comma or a closing square bracket

s_mov_b64 s[10:11], s[1:0]
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: first register index should not exceed second index
