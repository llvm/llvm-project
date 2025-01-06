// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=NOGFX10 --implicit-check-not=error: %s

buffer_atomic_add v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_x2 v[5:8], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_smax v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_smax_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_smin v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_smin_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_umax v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_umax_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_umin v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_umin_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor v5, v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_x2 v[5:6], v0, null, s3 idxen
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_d16_x v3, v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v3, v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[3:4], v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[3:5], v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[3:6], v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_dword v5, v0, null, s3 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_dwordx2 v[5:6], v0, null, s3 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_dwordx3 v[5:7], v0, null, s3 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_dwordx4 v[5:8], v0, null, s3 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_sbyte v5, v0, null, s3 idxen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_sshort v5, v0, null, s3 idxen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_ubyte v5, v0, null, s3 idxen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_ushort v5, v0, null, s3 idxen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_byte v1, v0, null, s4 idxen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_dword v1, v0, null, s4 idxen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_dwordx2 v[1:2], v0, null, s4 idxen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_dwordx3 v[1:3], v0, null, s4 idxen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_dwordx4 v[1:4], v0, null, s4 idxen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_d16_hi_x v1, v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_d16_x v1, v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_d16_xy v1, v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_d16_xyz v[1:2], v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_d16_xyzw v[1:3], v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], v0, null, s1 offen offset:4095
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
