// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefixes=NOGFX11 --implicit-check-not=error: %s

s_buffer_load_b32 s4, null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_b64 s4, null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_b128 s4, null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_b256 s4, null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_b512 s4, null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dword s4, null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dwordx2 s[4:5], null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dwordx4 s[4:7], null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dwordx8 s[4:11], null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dwordx16 s[4:19], null, s101
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_atc_probe_buffer 7, null, s2
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
