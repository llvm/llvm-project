// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s | FileCheck --check-prefix=GFX12 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: %s

tbuffer_load_format_d16_x v4, off, s[8:11], s3 format:[BUF_FMT_8_UNORM] offset:8388607
// GFX12: encoding: [0x03,0x00,0x22,0xc4,0x04,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v255, off, s[8:11], s3 format:1 offset:8388607
// GFX12: encoding: [0x03,0x00,0x22,0xc4,0xff,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX12: encoding: [0x03,0x00,0x22,0xc4,0x04,0x18,0x80,0x00,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, off, s[12:15], s101 format:[BUF_FMT_8_SNORM] offset:8388607
// GFX12: encoding: [0x65,0x00,0x22,0xc4,0x04,0x18,0x00,0x01,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, off, s[12:15], m0 format:2 offset:8388607
// GFX12: encoding: [0x7d,0x00,0x22,0xc4,0x04,0x18,0x00,0x01,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SNORM] offset:8388607
// GFX12: encoding: [0x00,0x00,0x22,0xc4,0x04,0x10,0x00,0x01,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, off, s[8:11], s61 format:[BUF_FMT_8_USCALED] offset:8388607
// GFX12: encoding: [0x3d,0x00,0x22,0xc4,0x04,0x10,0x80,0x01,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s61 format:3 offset:8388607
// GFX12: encoding: [0x3d,0x00,0x22,0xc4,0x04,0xe0,0x80,0x01,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX12: encoding: [0x03,0x00,0x22,0xc4,0x04,0x10,0x80,0x41,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_x v4, v1, s[8:11], s3 format:[BUF_FMT_8_SSCALED] idxen offset:52
// GFX12: encoding: [0x03,0x00,0x22,0xc4,0x04,0x10,0x00,0x82,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_x v4, v[1:2], s[8:11], s0 format:4 idxen offen offset:52
// GFX12: encoding: [0x00,0x00,0x22,0xc4,0x04,0x10,0x00,0xc2,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x00,0x22,0xc4,0x04,0xe0,0x68,0x02,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_UINT] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x00,0x22,0xc4,0x04,0xe0,0xbc,0x02,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3 format:5 offset:8388607 scope:SCOPE_SE
// GFX12: encoding: [0x03,0x00,0x22,0xc4,0x04,0xe0,0x84,0x02,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_SINT] offset:8388607
// GFX12: encoding: [0x03,0x40,0x22,0xc4,0x04,0x10,0x00,0x03,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v255, off, s[8:11], s3 format:6 offset:8388607
// GFX12: encoding: [0x03,0x40,0x22,0xc4,0xff,0x10,0x00,0x03,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX12: encoding: [0x03,0x40,0x22,0xc4,0x04,0x18,0x00,0x03,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, s[12:15], s101 format:[BUF_FMT_16_UNORM] offset:8388607
// GFX12: encoding: [0x65,0x40,0x22,0xc4,0x04,0x18,0x80,0x03,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, s[12:15], m0 format:7 offset:8388607
// GFX12: encoding: [0x7d,0x40,0x22,0xc4,0x04,0x18,0x80,0x03,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX12: encoding: [0x00,0x40,0x22,0xc4,0x04,0x10,0x80,0x03,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, s[8:11], s61 format:[BUF_FMT_16_SNORM] offset:8388607
// GFX12: encoding: [0x3d,0x40,0x22,0xc4,0x04,0x10,0x00,0x04,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s61 format:8 offset:8388607
// GFX12: encoding: [0x3d,0x40,0x22,0xc4,0x04,0xe0,0x00,0x04,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX12: encoding: [0x03,0x40,0x22,0xc4,0x04,0x10,0x00,0x44,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_xy v4, v1, s[8:11], s3 format:[BUF_FMT_16_USCALED] idxen offset:52
// GFX12: encoding: [0x03,0x40,0x22,0xc4,0x04,0x10,0x80,0x84,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_xy v4, v[1:2], s[8:11], s0 format:9 idxen offen offset:52
// GFX12: encoding: [0x00,0x40,0x22,0xc4,0x04,0x10,0x80,0xc4,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_USCALED] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x40,0x22,0xc4,0x04,0xe0,0xe8,0x04,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_FMT_16_SSCALED] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x40,0x22,0xc4,0x04,0xe0,0x3c,0x05,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3 format:10 offset:8388607 scope:SCOPE_SE
// GFX12: encoding: [0x03,0x40,0x22,0xc4,0x04,0xe0,0x04,0x05,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UINT] offset:8388607
// GFX12: encoding: [0x03,0x80,0x22,0xc4,0x04,0x10,0x80,0x05,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[254:255], off, s[8:11], s3 format:11 offset:8388607
// GFX12: encoding: [0x03,0x80,0x22,0xc4,0xfe,0x10,0x80,0x05,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, s[12:15], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX12: encoding: [0x03,0x80,0x22,0xc4,0x04,0x18,0x80,0x05,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, s[12:15], s101 format:[BUF_FMT_16_SINT] offset:8388607
// GFX12: encoding: [0x65,0x80,0x22,0xc4,0x04,0x18,0x00,0x06,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, s[12:15], m0 format:12 offset:8388607
// GFX12: encoding: [0x7d,0x80,0x22,0xc4,0x04,0x18,0x00,0x06,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, s[8:11], s0 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX12: encoding: [0x00,0x80,0x22,0xc4,0x04,0x10,0x00,0x06,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, s[8:11], s61 format:[BUF_FMT_16_FLOAT] offset:8388607
// GFX12: encoding: [0x3d,0x80,0x22,0xc4,0x04,0x10,0x80,0x06,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s61 format:13 offset:8388607
// GFX12: encoding: [0x3d,0x80,0x22,0xc4,0x04,0xe0,0x80,0x06,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_FLOAT] offen offset:52
// GFX12: encoding: [0x03,0x80,0x22,0xc4,0x04,0x10,0x80,0x46,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_xyz v[4:5], v1, s[8:11], s3 format:[BUF_FMT_8_8_UNORM] idxen offset:52
// GFX12: encoding: [0x03,0x80,0x22,0xc4,0x04,0x10,0x00,0x87,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_xyz v[4:5], v[1:2], s[8:11], s0 format:14 idxen offen offset:52
// GFX12: encoding: [0x00,0x80,0x22,0xc4,0x04,0x10,0x00,0xc7,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_UNORM] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x80,0x22,0xc4,0x04,0xe0,0x68,0x07,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_8_8_SNORM] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x80,0x22,0xc4,0x04,0xe0,0xbc,0x07,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:15 offset:8388607 scope:SCOPE_SE
// GFX12: encoding: [0x03,0x80,0x22,0xc4,0x04,0xe0,0x84,0x07,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_8_8_USCALED] offset:8388607
// GFX12: encoding: [0x03,0xc0,0x22,0xc4,0x04,0x10,0x00,0x08,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[254:255], off, s[8:11], s3 format:16 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x22,0xc4,0xfe,0x10,0x00,0x08,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, s[12:15], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_USCALED] offset:8388607
// GFX12: encoding: [0x03,0xc0,0x22,0xc4,0x04,0x18,0x00,0x08,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, s[12:15], s101 format:[BUF_FMT_8_8_SSCALED] offset:8388607
// GFX12: encoding: [0x65,0xc0,0x22,0xc4,0x04,0x18,0x80,0x08,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, s[12:15], m0 format:17 offset:8388607
// GFX12: encoding: [0x7d,0xc0,0x22,0xc4,0x04,0x18,0x80,0x08,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, s[8:11], s0 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607
// GFX12: encoding: [0x00,0xc0,0x22,0xc4,0x04,0x10,0x80,0x08,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, s[8:11], s61 format:[BUF_FMT_8_8_UINT] offset:8388607
// GFX12: encoding: [0x3d,0xc0,0x22,0xc4,0x04,0x10,0x00,0x09,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s61 format:18 offset:8388607
// GFX12: encoding: [0x3d,0xc0,0x22,0xc4,0x04,0xe0,0x00,0x09,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX12: encoding: [0x03,0xc0,0x22,0xc4,0x04,0x10,0x00,0x49,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_xyzw v[4:5], v1, s[8:11], s3 format:[BUF_FMT_8_8_SINT] idxen offset:52
// GFX12: encoding: [0x03,0xc0,0x22,0xc4,0x04,0x10,0x80,0x89,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_xyzw v[4:5], v[1:2], s[8:11], s0 format:19 idxen offen offset:52
// GFX12: encoding: [0x00,0xc0,0x22,0xc4,0x04,0x10,0x80,0xc9,0x01,0x34,0x00,0x00]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SINT] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0xc0,0x22,0xc4,0x04,0xe0,0xe8,0x09,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_UINT] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0xc0,0x22,0xc4,0x04,0xe0,0x3c,0x0a,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:20 offset:8388607 scope:SCOPE_SE
// GFX12: encoding: [0x03,0xc0,0x22,0xc4,0x04,0xe0,0x04,0x0a,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_x v4, off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SINT] offset:8388607
// GFX12: encoding: [0x03,0x00,0x23,0xc4,0x04,0x10,0x80,0x14,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v255, off, s[8:11], s3 format:41 offset:8388607
// GFX12: encoding: [0x03,0x00,0x23,0xc4,0xff,0x10,0x80,0x14,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX12: encoding: [0x03,0x00,0x23,0xc4,0x04,0x18,0x80,0x14,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, s[12:15], s101 format:[BUF_FMT_8_8_8_8_UNORM] offset:8388607
// GFX12: encoding: [0x65,0x00,0x23,0xc4,0x04,0x18,0x00,0x15,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, s[12:15], m0 format:42 offset:8388607
// GFX12: encoding: [0x7d,0x00,0x23,0xc4,0x04,0x18,0x00,0x15,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX12: encoding: [0x00,0x00,0x23,0xc4,0x04,0x10,0x00,0x15,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, s[8:11], s61 format:[BUF_FMT_8_8_8_8_SNORM] offset:8388607
// GFX12: encoding: [0x3d,0x00,0x23,0xc4,0x04,0x10,0x80,0x15,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s61 format:43 offset:8388607
// GFX12: encoding: [0x3d,0x00,0x23,0xc4,0x04,0xe0,0x80,0x15,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX12: encoding: [0x03,0x00,0x23,0xc4,0x04,0x10,0x80,0x55,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_x v4, v1, s[8:11], s3 format:[BUF_FMT_8_8_8_8_USCALED] idxen offset:52
// GFX12: encoding: [0x03,0x00,0x23,0xc4,0x04,0x10,0x00,0x96,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_x v4, v[1:2], s[8:11], s0 format:44 idxen offen offset:52
// GFX12: encoding: [0x00,0x00,0x23,0xc4,0x04,0x10,0x00,0xd6,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_USCALED] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x00,0x23,0xc4,0x04,0xe0,0x68,0x16,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_8_8_8_SSCALED] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x00,0x23,0xc4,0x04,0xe0,0xbc,0x16,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3 format:45 offset:8388607 scope:SCOPE_SE
// GFX12: encoding: [0x03,0x00,0x23,0xc4,0x04,0xe0,0x84,0x16,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_8_8_8_UINT] offset:8388607
// GFX12: encoding: [0x03,0x40,0x23,0xc4,0x04,0x10,0x00,0x17,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v255, off, s[8:11], s3 format:46 offset:8388607
// GFX12: encoding: [0x03,0x40,0x23,0xc4,0xff,0x10,0x00,0x17,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX12: encoding: [0x03,0x40,0x23,0xc4,0x04,0x18,0x00,0x17,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, s[12:15], s101 format:[BUF_FMT_8_8_8_8_SINT] offset:8388607
// GFX12: encoding: [0x65,0x40,0x23,0xc4,0x04,0x18,0x80,0x17,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, s[12:15], m0 format:47 offset:8388607
// GFX12: encoding: [0x7d,0x40,0x23,0xc4,0x04,0x18,0x80,0x17,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX12: encoding: [0x00,0x40,0x23,0xc4,0x04,0x10,0x80,0x17,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, s[8:11], s61 format:[BUF_FMT_32_32_UINT] offset:8388607
// GFX12: encoding: [0x3d,0x40,0x23,0xc4,0x04,0x10,0x00,0x18,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s61 format:48 offset:8388607
// GFX12: encoding: [0x3d,0x40,0x23,0xc4,0x04,0xe0,0x00,0x18,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX12: encoding: [0x03,0x40,0x23,0xc4,0x04,0x10,0x00,0x58,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_xy v4, v1, s[8:11], s3 format:[BUF_FMT_32_32_SINT] idxen offset:52
// GFX12: encoding: [0x03,0x40,0x23,0xc4,0x04,0x10,0x80,0x98,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_xy v4, v[1:2], s[8:11], s0 format:49 idxen offen offset:52
// GFX12: encoding: [0x00,0x40,0x23,0xc4,0x04,0x10,0x80,0xd8,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_SINT] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x40,0x23,0xc4,0x04,0xe0,0xe8,0x18,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_FMT_32_32_FLOAT] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x40,0x23,0xc4,0x04,0xe0,0x3c,0x19,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3 format:50 offset:8388607 scope:SCOPE_SE
// GFX12: encoding: [0x03,0x40,0x23,0xc4,0x04,0xe0,0x04,0x19,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607
// GFX12: encoding: [0x03,0x80,0x23,0xc4,0x04,0x10,0x80,0x19,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[254:255], off, s[8:11], s3 format:51 offset:8388607
// GFX12: encoding: [0x03,0x80,0x23,0xc4,0xfe,0x10,0x80,0x19,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, s[12:15], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX12: encoding: [0x03,0x80,0x23,0xc4,0x04,0x18,0x80,0x19,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, s[12:15], s101 format:[BUF_FMT_16_16_16_16_SNORM] offset:8388607
// GFX12: encoding: [0x65,0x80,0x23,0xc4,0x04,0x18,0x00,0x1a,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, s[12:15], m0 format:52 offset:8388607
// GFX12: encoding: [0x7d,0x80,0x23,0xc4,0x04,0x18,0x00,0x1a,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, s[8:11], s0 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SNORM] offset:8388607
// GFX12: encoding: [0x00,0x80,0x23,0xc4,0x04,0x10,0x00,0x1a,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, s[8:11], s61 format:[BUF_FMT_16_16_16_16_USCALED] offset:8388607
// GFX12: encoding: [0x3d,0x80,0x23,0xc4,0x04,0x10,0x80,0x1a,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s61 format:53 offset:8388607
// GFX12: encoding: [0x3d,0x80,0x23,0xc4,0x04,0xe0,0x80,0x1a,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX12: encoding: [0x03,0x80,0x23,0xc4,0x04,0x10,0x80,0x5a,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_xyz v[4:5], v1, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SSCALED] idxen offset:52
// GFX12: encoding: [0x03,0x80,0x23,0xc4,0x04,0x10,0x00,0x9b,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_xyz v[4:5], v[1:2], s[8:11], s0 format:54 idxen offen offset:52
// GFX12: encoding: [0x00,0x80,0x23,0xc4,0x04,0x10,0x00,0xdb,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0x80,0x23,0xc4,0x04,0xe0,0x68,0x1b,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_16_16_16_16_UINT] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0x80,0x23,0xc4,0x04,0xe0,0xbc,0x1b,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:55 offset:8388607 scope:SCOPE_SE
// GFX12: encoding: [0x03,0x80,0x23,0xc4,0x04,0xe0,0x84,0x1b,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SINT] offset:8388607
// GFX12: encoding: [0x03,0xc0,0x23,0xc4,0x04,0x10,0x00,0x1c,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[254:255], off, s[8:11], s3 format:56 offset:8388607
// GFX12: encoding: [0x03,0xc0,0x23,0xc4,0xfe,0x10,0x00,0x1c,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, s[12:15], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX12: encoding: [0x03,0xc0,0x23,0xc4,0x04,0x18,0x00,0x1c,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, s[12:15], s101 format:[BUF_FMT_16_16_16_16_FLOAT] offset:8388607
// GFX12: encoding: [0x65,0xc0,0x23,0xc4,0x04,0x18,0x80,0x1c,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, s[12:15], m0 format:57 offset:8388607
// GFX12: encoding: [0x7d,0xc0,0x23,0xc4,0x04,0x18,0x80,0x1c,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, s[8:11], s0 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX12: encoding: [0x00,0xc0,0x23,0xc4,0x04,0x10,0x80,0x1c,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, s[8:11], s61 format:[BUF_FMT_32_32_32_UINT] offset:8388607
// GFX12: encoding: [0x3d,0xc0,0x23,0xc4,0x04,0x10,0x00,0x1d,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s61 format:58 offset:8388607
// GFX12: encoding: [0x3d,0xc0,0x23,0xc4,0x04,0xe0,0x00,0x1d,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX12: encoding: [0x03,0xc0,0x23,0xc4,0x04,0x10,0x00,0x5d,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_xyzw v[4:5], v1, s[8:11], s3 format:[BUF_FMT_32_32_32_SINT] idxen offset:52
// GFX12: encoding: [0x03,0xc0,0x23,0xc4,0x04,0x10,0x80,0x9d,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_xyzw v[4:5], v[1:2], s[8:11], s0 format:59 idxen offen offset:52
// GFX12: encoding: [0x00,0xc0,0x23,0xc4,0x04,0x10,0x80,0xdd,0x01,0x34,0x00,0x00]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_SINT] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX12: encoding: [0x03,0xc0,0x23,0xc4,0x04,0xe0,0xe8,0x1d,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_32_32_FLOAT] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX12: encoding: [0x03,0xc0,0x23,0xc4,0x04,0xe0,0x3c,0x1e,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:60 offset:8388607 scope:SCOPE_SE
// GFX12: encoding: [0x03,0xc0,0x23,0xc4,0x04,0xe0,0x04,0x1e,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 glc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 slc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 dlc
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
