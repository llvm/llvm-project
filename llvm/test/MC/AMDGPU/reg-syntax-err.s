// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOVI --implicit-check-not=error: %s

s_mov_b32 s1, s 1
// NOVI: error: invalid operand for instruction

s_mov_b32 s1, s[0 1
// NOVI: error: expected a closing square bracket

s_mov_b32 s1, s[0:0 1
// NOVI: error: expected a closing square bracket

s_mov_b32 s1, [s[0 1
// NOVI: error: expected a closing square bracket

s_mov_b32 s1, [s[0:1] 1
// NOVI: error: expected a single 32-bit register

s_mov_b32 s1, [s0, 1
// NOVI: error: expected a register or a list of registers

s_mov_b32 s1, s999 1
// NOVI: error: register index is out of range

s_mov_b32 s1, s[1:2] 1
// NOVI: error: invalid register alignment

s_mov_b32 s1, s[0:2] 1
// NOVI: error: invalid operand for instruction

s_mov_b32 s1, xnack_mask_lo 1
// NOVI: error: register not available on this GPU

s_mov_b32 s1, s s0
// NOVI: error: invalid operand for instruction

s_mov_b32 s1, s[0 s0
// NOVI: error: expected a closing square bracket

s_mov_b32 s1, s[0:0 s0
// NOVI: error: expected a closing square bracket

s_mov_b32 s1, [s[0 s0
// NOVI: error: expected a closing square bracket

s_mov_b32 s1, [s[0:1] s0
// NOVI: error: expected a single 32-bit register

s_mov_b32 s1, [s0, s0
// NOVI: error: registers in a list must have consecutive indices

s_mov_b32 s1, s999 s0
// NOVI: error: register index is out of range

s_mov_b32 s1, s[1:2] s0
// NOVI: error: invalid register alignment

s_mov_b32 s1, s[0:2] vcc_lo
// NOVI: error: invalid operand for instruction

s_mov_b32 s1, xnack_mask_lo s1
// NOVI: error: register not available on this GPU

exp mrt0 v1, v2, v3, v4000 off
// NOVI: error: register index is out of range

v_add_f64 v[0:1], v[0:1], v[0xF00000001:0x2]
// NOVI: error: invalid register index

v_add_f64 v[0:1], v[0:1], v[0x1:0xF00000002]
// NOVI: error: invalid register index

s_mov_b32 s1, s[0:-1]
// NOVI: error: invalid register index

s_mov_b64 s[10:11], [exec_lo,vcc_hi]
// NOVI: error: register does not fit in the list

s_mov_b64 s[10:11], [exec_hi,exec_lo]
// NOVI: error: register does not fit in the list

s_mov_b64 s[10:11], [exec_lo,exec_lo]
// NOVI: error: register does not fit in the list

s_mov_b64 s[10:11], [exec,exec_lo]
// NOVI: error: register does not fit in the list

s_mov_b64 s[10:11], [exec_lo,exec]
// NOVI: error: register does not fit in the list

s_mov_b64 s[10:11], [exec_lo,s0]
// NOVI: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s0,exec_lo]
// NOVI: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s0,exec]
// NOVI: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s0,v1]
// NOVI: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [v0,s1]
// NOVI: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s0,s0]
// NOVI: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [s0,s2]
// NOVI: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [s2,s1]
// NOVI: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [a0,a2]
// NOVI: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [a0,v1]
// NOVI: error: registers in a list must be of the same kind

s_mov_b64 s[10:11], [s
// NOVI: error: missing register index

s_mov_b64 s[10:11], s[1:0]
// NOVI: error: first register index should not exceed second index

s_mov_b64 s[10:11], [x0,s1]
// NOVI: error: invalid register name

s_mov_b64 s[10:11], [s,s1]
// NOVI: error: missing register index

s_mov_b64 s[10:11], [s01,s1]
// NOVI: error: registers in a list must have consecutive indices

s_mov_b64 s[10:11], [s0x]
// NOVI: error: invalid register index

s_mov_b64 s[10:11], [s[0:1],s[2:3]]
// NOVI: error: expected a single 32-bit register

s_mov_b64 s[10:11], [s0,s[2:3]]
// NOVI: error: expected a single 32-bit register

s_mov_b64 s[10:11], [s0
// NOVI: error: expected a comma or a closing square bracket

s_mov_b64 s[10:11], [s0,s1
// NOVI: error: expected a comma or a closing square bracket

s_mov_b64 s[10:11], s[1:0]
// NOVI: error: first register index should not exceed second index
