// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s | FileCheck --check-prefix=GFX12 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: %s

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_x v255, off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0xff,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_x v5, off, s[12:15], s3 offset:8388607
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_x v5, off, s[96:99], s3 offset:8388607
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_x v5, off, s[8:11], s101 offset:8388607
// GFX12: encoding: [0x65,0x00,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_x v5, off, s[8:11], m0 offset:8388607
// GFX12: encoding: [0x7d,0x00,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_x v5, off, s[8:11], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_x v5, off, s[8:11], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_x v5, off, s[8:11], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_x v5, off, s[8:11], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_x v5, v0, s[8:11], s3 idxen offset:8388607
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0x10,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_x v5, v0, s[8:11], s3 offen offset:8388607
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0x10,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_x v5, off, s[8:11], s3
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:0
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:7
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0x10,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x00,0x02,0xc4,0x05,0x10,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v255, off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0xff,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v5, off, s[12:15], s3 offset:8388607
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v5, off, s[96:99], s3 offset:8388607
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v5, off, s[8:11], s101 offset:8388607
// GFX12: encoding: [0x65,0x40,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v5, off, s[8:11], m0 offset:8388607
// GFX12: encoding: [0x7d,0x40,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v5, off, s[8:11], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xy v5, off, s[8:11], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xy v5, off, s[8:11], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xy v5, off, s[8:11], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xy v5, v0, s[8:11], s3 idxen offset:8388607
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0x10,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v5, v0, s[8:11], s3 offen offset:8388607
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0x10,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v5, off, s[8:11], s3
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:0
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:7
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0x10,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x40,0x02,0xc4,0x05,0x10,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[254:255], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0xfe,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[5:6], off, s[12:15], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[5:6], off, s[96:99], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s101 offset:8388607
// GFX12: encoding: [0x65,0x80,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[5:6], off, s[8:11], m0 offset:8388607
// GFX12: encoding: [0x7d,0x80,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[5:6], off, s[8:11], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyz v[5:6], off, s[8:11], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyz v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyz v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyz v[5:6], v0, s[8:11], s3 idxen offset:8388607
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0x10,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[5:6], v0, s[8:11], s3 offen offset:8388607
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0x10,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:0
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:7
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0x10,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x80,0x02,0xc4,0x05,0x10,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[254:255], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0xfe,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[5:6], off, s[12:15], s3 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[5:6], off, s[96:99], s3 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s101 offset:8388607
// GFX12: encoding: [0x65,0xc0,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], m0 offset:8388607
// GFX12: encoding: [0x7d,0xc0,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyzw v[5:6], v0, s[8:11], s3 idxen offset:8388607
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0x10,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[5:6], v0, s[8:11], s3 offen offset:8388607
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0x10,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:0
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:7
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0x10,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0x10,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0xc0,0x02,0xc4,0x05,0x10,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v255, off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0xff,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v5, off, s[12:15], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v5, off, s[96:99], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v5, off, s[8:11], s101 offset:8388607
// GFX12: encoding: [0x65,0x80,0x09,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v5, off, s[8:11], m0 offset:8388607
// GFX12: encoding: [0x7d,0x80,0x09,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v5, off, s[8:11], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_format_x v5, off, s[8:11], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_format_x v5, off, s[8:11], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_format_x v5, off, s[8:11], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_format_x v5, v0, s[8:11], s3 idxen offset:8388607
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0x10,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v5, v0, s[8:11], s3 offen offset:8388607
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0x10,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v5, off, s[8:11], s3
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:0
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:7
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0x10,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0x10,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x80,0x09,0xc4,0x05,0x10,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_x v255, off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0xff,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_x v5, off, s[12:15], s3 offset:8388607
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_x v5, off, s[96:99], s3 offset:8388607
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_x v5, off, s[8:11], s101 offset:8388607
// GFX12: encoding: [0x65,0x00,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_x v5, off, s[8:11], m0 offset:8388607
// GFX12: encoding: [0x7d,0x00,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_x v5, off, s[8:11], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_x v5, off, s[8:11], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_x v5, off, s[8:11], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_x v5, off, s[8:11], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_x v5, v0, s[8:11], s3 idxen offset:8388607
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0x10,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_load_format_x v5, v0, s[8:11], s3 offen offset:8388607
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0x10,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_load_format_x v5, off, s[8:11], s3
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_format_x v5, off, s[8:11], s3 offset:0
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_format_x v5, off, s[8:11], s3 offset:7
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0x10,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x00,0x00,0xc4,0x05,0x10,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[254:255], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0xfe,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:6], off, s[12:15], s3 offset:8388607
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:6], off, s[96:99], s3 offset:8388607
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:6], off, s[8:11], s101 offset:8388607
// GFX12: encoding: [0x65,0x40,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:6], off, s[8:11], m0 offset:8388607
// GFX12: encoding: [0x7d,0x40,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:6], off, s[8:11], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xy v[5:6], off, s[8:11], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xy v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xy v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xy v[5:6], v0, s[8:11], s3 idxen offset:8388607
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0x10,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:6], v0, s[8:11], s3 offen offset:8388607
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0x10,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:6], off, s[8:11], s3
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:0
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:7
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0x10,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x40,0x00,0xc4,0x05,0x10,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xy v[5:7], off, s[8:11], s3 offset:8388607 glc slc dlc tfe
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyz v[253:255], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0xfd,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyz v[5:7], off, s[12:15], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyz v[5:7], off, s[96:99], s3 offset:8388607
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyz v[5:7], off, s[8:11], s101 offset:8388607
// GFX12: encoding: [0x65,0x80,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyz v[5:7], off, s[8:11], m0 offset:8388607
// GFX12: encoding: [0x7d,0x80,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyz v[5:7], off, s[8:11], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyz v[5:7], off, s[8:11], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyz v[5:7], off, s[8:11], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyz v[5:7], off, s[8:11], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyz v[5:7], v0, s[8:11], s3 idxen offset:8388607
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0x10,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_load_format_xyz v[5:7], v0, s[8:11], s3 offen offset:8388607
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0x10,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_load_format_xyz v[5:7], off, s[8:11], s3
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:0
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:7
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0x10,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x80,0x00,0xc4,0x05,0x10,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[252:255], off, s[8:11], s3 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0xfc,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[5:8], off, s[12:15], s3 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[5:8], off, s[96:99], s3 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[5:8], off, s[8:11], s101 offset:8388607
// GFX12: encoding: [0x65,0xc0,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[5:8], off, s[8:11], m0 offset:8388607
// GFX12: encoding: [0x7d,0xc0,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[5:8], off, s[8:11], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyzw v[5:8], off, s[8:11], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyzw v[5:8], off, s[8:11], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyzw v[5:8], off, s[8:11], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyzw v[5:8], v0, s[8:11], s3 idxen offset:8388607
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0x10,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[5:8], v0, s[8:11], s3 offen offset:8388607
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0x10,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[5:8], off, s[8:11], s3
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:0
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:7
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0x10,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0x10,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0xc0,0x00,0xc4,0x05,0x10,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v255, off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0xff,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v1, off, s[16:19], s4 offset:8388607
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0x20,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v1, off, s[96:99], s4 offset:8388607
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v1, off, s[12:15], s101 offset:8388607
// GFX12: encoding: [0x65,0x00,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v1, off, s[12:15], m0 offset:8388607
// GFX12: encoding: [0x7d,0x00,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v1, off, s[12:15], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_x v1, off, s[12:15], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_x v1, off, s[12:15], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_x v1, off, s[12:15], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_x v1, v0, s[12:15], s4 idxen offset:8388607
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v1, v0, s[12:15], s4 offen offset:8388607
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0x18,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v1, off, s[12:15], s4
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:0
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:7
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0x18,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x04,0x00,0x03,0xc4,0x01,0x18,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v255, off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0xff,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v1, off, s[16:19], s4 offset:8388607
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0x20,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v1, off, s[96:99], s4 offset:8388607
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v1, off, s[12:15], s101 offset:8388607
// GFX12: encoding: [0x65,0x40,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v1, off, s[12:15], m0 offset:8388607
// GFX12: encoding: [0x7d,0x40,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v1, off, s[12:15], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xy v1, off, s[12:15], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xy v1, off, s[12:15], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xy v1, off, s[12:15], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xy v1, v0, s[12:15], s4 idxen offset:8388607
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v1, v0, s[12:15], s4 offen offset:8388607
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0x18,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v1, off, s[12:15], s4
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:0
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:7
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0x18,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x04,0x40,0x03,0xc4,0x01,0x18,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[254:255], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0xfe,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[1:2], off, s[16:19], s4 offset:8388607
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0x20,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[1:2], off, s[96:99], s4 offset:8388607
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s101 offset:8388607
// GFX12: encoding: [0x65,0x80,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[1:2], off, s[12:15], m0 offset:8388607
// GFX12: encoding: [0x7d,0x80,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[1:2], off, s[12:15], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyz v[1:2], off, s[12:15], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyz v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyz v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyz v[1:2], v0, s[12:15], s4 idxen offset:8388607
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[1:2], v0, s[12:15], s4 offen offset:8388607
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0x18,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:0
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:7
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0x18,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x04,0x80,0x03,0xc4,0x01,0x18,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[254:255], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0xfe,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[1:2], off, s[16:19], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0x20,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[1:2], off, s[96:99], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s101 offset:8388607
// GFX12: encoding: [0x65,0xc0,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], m0 offset:8388607
// GFX12: encoding: [0x7d,0xc0,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyzw v[1:2], v0, s[12:15], s4 idxen offset:8388607
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[1:2], v0, s[12:15], s4 offen offset:8388607
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0x18,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:0
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:7
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0x18,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0x18,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x04,0xc0,0x03,0xc4,0x01,0x18,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v255, off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0xff,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v1, off, s[16:19], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0x20,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v1, off, s[96:99], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v1, off, s[12:15], s101 offset:8388607
// GFX12: encoding: [0x65,0xc0,0x09,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v1, off, s[12:15], m0 offset:8388607
// GFX12: encoding: [0x7d,0xc0,0x09,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v1, off, s[12:15], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_format_x v1, off, s[12:15], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_format_x v1, off, s[12:15], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_format_x v1, off, s[12:15], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_format_x v1, v0, s[12:15], s4 idxen offset:8388607
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v1, v0, s[12:15], s4 offen offset:8388607
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0x18,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v1, off, s[12:15], s4
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:0
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:7
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0x18,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0x18,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x04,0xc0,0x09,0xc4,0x01,0x18,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_x v255, off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0xff,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_x v1, off, s[16:19], s4 offset:8388607
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0x20,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_x v1, off, s[96:99], s4 offset:8388607
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_x v1, off, s[12:15], s101 offset:8388607
// GFX12: encoding: [0x65,0x00,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_x v1, off, s[12:15], m0 offset:8388607
// GFX12: encoding: [0x7d,0x00,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_x v1, off, s[12:15], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_x v1, off, s[12:15], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_x v1, off, s[12:15], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_x v1, off, s[12:15], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_x v1, v0, s[12:15], s4 idxen offset:8388607
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_store_format_x v1, v0, s[12:15], s4 offen offset:8388607
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0x18,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_store_format_x v1, off, s[12:15], s4
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_format_x v1, off, s[12:15], s4 offset:0
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_format_x v1, off, s[12:15], s4 offset:7
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0x18,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x04,0x00,0x01,0xc4,0x01,0x18,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[254:255], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0xfe,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[1:2], off, s[16:19], s4 offset:8388607
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0x20,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[1:2], off, s[96:99], s4 offset:8388607
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[1:2], off, s[12:15], s101 offset:8388607
// GFX12: encoding: [0x65,0x40,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[1:2], off, s[12:15], m0 offset:8388607
// GFX12: encoding: [0x7d,0x40,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[1:2], off, s[12:15], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xy v[1:2], off, s[12:15], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xy v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xy v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xy v[1:2], v0, s[12:15], s4 idxen offset:8388607
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[1:2], v0, s[12:15], s4 offen offset:8388607
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0x18,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[1:2], off, s[12:15], s4
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:0
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:7
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0x18,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x04,0x40,0x01,0xc4,0x01,0x18,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[253:255], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0xfd,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[1:3], off, s[16:19], s4 offset:8388607
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0x20,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[1:3], off, s[96:99], s4 offset:8388607
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[1:3], off, s[12:15], s101 offset:8388607
// GFX12: encoding: [0x65,0x80,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[1:3], off, s[12:15], m0 offset:8388607
// GFX12: encoding: [0x7d,0x80,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[1:3], off, s[12:15], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyz v[1:3], off, s[12:15], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyz v[1:3], off, s[12:15], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyz v[1:3], off, s[12:15], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyz v[1:3], v0, s[12:15], s4 idxen offset:8388607
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[1:3], v0, s[12:15], s4 offen offset:8388607
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0x18,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[1:3], off, s[12:15], s4
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:0
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:7
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0x18,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x04,0x80,0x01,0xc4,0x01,0x18,0xbc,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[252:255], off, s[12:15], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0xfc,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[1:4], off, s[16:19], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0x20,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[1:4], off, s[96:99], s4 offset:8388607
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0xc0,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[1:4], off, s[12:15], s101 offset:8388607
// GFX12: encoding: [0x65,0xc0,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[1:4], off, s[12:15], m0 offset:8388607
// GFX12: encoding: [0x7d,0xc0,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[1:4], off, s[12:15], 0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyzw v[1:4], off, s[12:15], -1 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyzw v[1:4], off, s[12:15], 0.5 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyzw v[1:4], off, s[12:15], -4.0 offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyzw v[1:4], v0, s[12:15], s4 idxen offset:8388607
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[1:4], v0, s[12:15], s4 offen offset:8388607
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0x18,0x80,0x40,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[1:4], off, s[12:15], s4
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:0
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x00,0x00,0x00]

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:7
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0x18,0x80,0x00,0x00,0x07,0x00,0x00]

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0x18,0xe8,0x00,0x00,0xff,0xff,0x7f]

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x04,0xc0,0x01,0xc4,0x01,0x18,0xbc,0x00,0x00,0xff,0xff,0x7f]
