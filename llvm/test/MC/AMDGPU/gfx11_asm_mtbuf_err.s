// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefixes=NOGFX11 --implicit-check-not=error: %s

tbuffer_load_format_d16_x v3, v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xy v[3:4], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xyz v[3:5], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xyzw v[3:6], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_x v3, v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xy v[3:4], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xyz v[3:5], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xyzw v[3:6], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_x v3, v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xy v[3:4], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xyz v[3:5], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xyzw v[3:6], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_x v3, v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xy v[3:4], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xyz v[3:5], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xyzw v[3:6], v0, null, s1 offen offset:4095
// NOGFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
