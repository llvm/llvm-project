// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=NOGFX10 --implicit-check-not=error: %s

s_buffer_atomic_add s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_add_x2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_and s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_cmpswap s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_cmpswap_x2 s[4:7], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_dec s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_dec_x2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_inc s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_inc_x2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_or s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_or_x2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_smax s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_smax_x2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_smin s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_smin_x2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_sub s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_sub_x2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_swap s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_umax s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_umax_x2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_umin s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_atomic_umin_x2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dword s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dwordx2 s[4:5], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dwordx4 s[4:7], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dwordx8 s[4:11], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_load_dwordx16 s[4:19], null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_buffer_store_dword s4, null, s101
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_atc_probe_buffer 7, null, s2
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
