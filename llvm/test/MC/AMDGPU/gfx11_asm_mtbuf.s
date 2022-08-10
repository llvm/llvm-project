// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX11-ERR --implicit-check-not=error: %s

tbuffer_load_d16_format_x v4, off, s[8:11], s3, format:[BUF_FMT_8_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x0c,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_d16_format_x v255, off, s[8:11], s3, format:1 offset:4095
// GFX11: encoding: [0xff,0x0f,0x0c,0xe8,0x00,0xff,0x02,0x03]

tbuffer_load_d16_format_x v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x0c,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_d16_format_x v4, off, s[12:15], s101, format:[BUF_FMT_8_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x14,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_d16_format_x v4, off, s[12:15], m0, format:2 offset:4095
// GFX11: encoding: [0xff,0x0f,0x14,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_d16_format_x v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x14,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_d16_format_x v4, off, s[8:11], 61, format:[BUF_FMT_8_USCALED] offset:4095
// GFX11: encoding: [0xff,0x0f,0x1c,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_d16_format_x v4, off, ttmp[4:7], 61, format:3 offset:4095
// GFX11: encoding: [0xff,0x0f,0x1c,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_d16_format_x v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX11: encoding: [0x34,0x00,0x1c,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_d16_format_x v4, v1, s[8:11], s3, format:[BUF_FMT_8_SSCALED] idxen offset:52
// GFX11: encoding: [0x34,0x00,0x24,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_d16_format_x v4, v[1:2], s[8:11], s0, format:4 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0x24,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SSCALED] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0x24,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3, format:[BUF_FMT_8_UINT] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0x2c,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3, format:5 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0x2c,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0x2c,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xy v4, off, s[8:11], s3, format:[BUF_FMT_8_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x34,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_d16_format_xy v255, off, s[8:11], s3, format:6 offset:4095
// GFX11: encoding: [0xff,0x8f,0x34,0xe8,0x00,0xff,0x02,0x03]

tbuffer_load_d16_format_xy v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x34,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_d16_format_xy v4, off, s[12:15], s101, format:[BUF_FMT_16_UNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x3c,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_d16_format_xy v4, off, s[12:15], m0, format:7 offset:4095
// GFX11: encoding: [0xff,0x8f,0x3c,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_d16_format_xy v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x3c,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_d16_format_xy v4, off, s[8:11], 61, format:[BUF_FMT_16_SNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x44,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], 61, format:8 offset:4095
// GFX11: encoding: [0xff,0x8f,0x44,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_d16_format_xy v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX11: encoding: [0x34,0x80,0x44,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_d16_format_xy v4, v1, s[8:11], s3, format:[BUF_FMT_16_USCALED] idxen offset:52
// GFX11: encoding: [0x34,0x80,0x4c,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_d16_format_xy v4, v[1:2], s[8:11], s0, format:9 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0x4c,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_USCALED] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0x4c,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3, format:[BUF_FMT_16_SSCALED] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0x54,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3, format:10 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0x54,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0x54,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_UINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x5d,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_d16_format_xyz v[254:255], off, s[8:11], s3, format:11 offset:4095
// GFX11: encoding: [0xff,0x0f,0x5d,0xe8,0x00,0xfe,0x02,0x03]

tbuffer_load_d16_format_xyz v[4:5], off, s[12:15], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_UINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x5d,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_d16_format_xyz v[4:5], off, s[12:15], s101, format:[BUF_FMT_16_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x65,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_d16_format_xyz v[4:5], off, s[12:15], m0, format:12 offset:4095
// GFX11: encoding: [0xff,0x0f,0x65,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], 0, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x65,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], 61, format:[BUF_FMT_16_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x6d,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], 61, format:13 offset:4095
// GFX11: encoding: [0xff,0x0f,0x6d,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_d16_format_xyz v[4:5], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_FLOAT] offen offset:52
// GFX11: encoding: [0x34,0x00,0x6d,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_d16_format_xyz v[4:5], v1, s[8:11], s3, format:[BUF_FMT_8_8_UNORM] idxen offset:52
// GFX11: encoding: [0x34,0x00,0x75,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_d16_format_xyz v[4:5], v[1:2], s[8:11], s0, format:14 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0x75,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_UNORM] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0x75,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_8_8_SNORM] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0x7d,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3, format:15 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0x7d,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0x7d,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], s3, format:[BUF_FMT_8_8_USCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x85,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_d16_format_xyzw v[254:255], off, s[8:11], s3, format:16 offset:4095
// GFX11: encoding: [0xff,0x8f,0x85,0xe8,0x00,0xfe,0x02,0x03]

tbuffer_load_d16_format_xyzw v[4:5], off, s[12:15], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_USCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x85,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_d16_format_xyzw v[4:5], off, s[12:15], s101, format:[BUF_FMT_8_8_SSCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x8d,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_d16_format_xyzw v[4:5], off, s[12:15], m0, format:17 offset:4095
// GFX11: encoding: [0xff,0x8f,0x8d,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], 0, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SSCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x8d,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], 61, format:[BUF_FMT_8_8_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x95,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], 61, format:18 offset:4095
// GFX11: encoding: [0xff,0x8f,0x95,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_d16_format_xyzw v[4:5], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX11: encoding: [0x34,0x80,0x95,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_d16_format_xyzw v[4:5], v1, s[8:11], s3, format:[BUF_FMT_8_8_SINT] idxen offset:52
// GFX11: encoding: [0x34,0x80,0x9d,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_d16_format_xyzw v[4:5], v[1:2], s[8:11], s0, format:19 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0x9d,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SINT] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0x9d,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_32_UINT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0xa5,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3, format:20 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0xa5,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0xa5,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_32_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xa8,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_x v255, off, s[8:11], s3, format:21 offset:4095
// GFX11: encoding: [0xff,0x0f,0xa8,0xe8,0x00,0xff,0x02,0x03]

tbuffer_load_format_x v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xa8,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_format_x v4, off, s[12:15], s101, format:[BUF_FMT_32_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xb0,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_format_x v4, off, s[12:15], m0, format:22 offset:4095
// GFX11: encoding: [0xff,0x0f,0xb0,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_format_x v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xb0,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_format_x v4, off, s[8:11], 61, format:[BUF_FMT_16_16_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0xb8,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_format_x v4, off, ttmp[4:7], 61, format:23 offset:4095
// GFX11: encoding: [0xff,0x0f,0xb8,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_format_x v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_UNORM] offen offset:52
// GFX11: encoding: [0x34,0x00,0xb8,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_format_x v4, v1, s[8:11], s3, format:[BUF_FMT_16_16_SNORM] idxen offset:52
// GFX11: encoding: [0x34,0x00,0xc0,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_format_x v4, v[1:2], s[8:11], s0, format:24 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0xc0,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_format_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_SNORM] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0xc0,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_x v4, off, ttmp[4:7], s3, format:[BUF_FMT_16_16_USCALED] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0xc8,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_x v4, off, ttmp[4:7], s3, format:25 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0xc8,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_USCALED] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0xc8,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xy v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_16_SSCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0xd0,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_xy v[254:255], off, s[8:11], s3, format:26 offset:4095
// GFX11: encoding: [0xff,0x8f,0xd0,0xe8,0x00,0xfe,0x02,0x03]

tbuffer_load_format_xy v[4:5], off, s[12:15], s3, format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_SSCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0xd0,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_format_xy v[4:5], off, s[12:15], s101, format:[BUF_FMT_16_16_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xd8,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_format_xy v[4:5], off, s[12:15], m0, format:27 offset:4095
// GFX11: encoding: [0xff,0x8f,0xd8,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_format_xy v[4:5], off, s[8:11], 0, format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xd8,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_format_xy v[4:5], off, s[8:11], 61, format:[BUF_FMT_16_16_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xe0,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], 61, format:28 offset:4095
// GFX11: encoding: [0xff,0x8f,0xe0,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_format_xy v[4:5], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_SINT] offen offset:52
// GFX11: encoding: [0x34,0x80,0xe0,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_format_xy v[4:5], v1, s[8:11], s3, format:[BUF_FMT_16_16_FLOAT] idxen offset:52
// GFX11: encoding: [0x34,0x80,0xe8,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_format_xy v[4:5], v[1:2], s[8:11], s0, format:29 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0xe8,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_FLOAT] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0xe8,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_10_11_11_FLOAT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0xf0,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3, format:30 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0xf0,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_10_11_11, BUF_NUM_FORMAT_FLOAT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0xf0,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xyz v[4:6], off, s[8:11], s3, format:[BUF_FMT_11_11_10_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xf9,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_xyz v[253:255], off, s[8:11], s3, format:31 offset:4095
// GFX11: encoding: [0xff,0x0f,0xf9,0xe8,0x00,0xfd,0x02,0x03]

tbuffer_load_format_xyz v[4:6], off, s[12:15], s3, format:[BUF_DATA_FORMAT_11_11_10, BUF_NUM_FORMAT_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xf9,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_format_xyz v[4:6], off, s[12:15], s101, format:[BUF_FMT_10_10_10_2_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x01,0xe9,0x00,0x04,0x03,0x65]

tbuffer_load_format_xyz v[4:6], off, s[12:15], m0, format:32 offset:4095
// GFX11: encoding: [0xff,0x0f,0x01,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_load_format_xyz v[4:6], off, s[8:11], 0, format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x01,0xe9,0x00,0x04,0x02,0x80]

tbuffer_load_format_xyz v[4:6], off, s[8:11], 61, format:[BUF_FMT_10_10_10_2_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x09,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], 61, format:33 offset:4095
// GFX11: encoding: [0xff,0x0f,0x09,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_load_format_xyz v[4:6], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX11: encoding: [0x34,0x00,0x09,0xe9,0x01,0x04,0x42,0x03]

tbuffer_load_format_xyz v[4:6], v1, s[8:11], s3, format:[BUF_FMT_10_10_10_2_UINT] idxen offset:52
// GFX11: encoding: [0x34,0x00,0x11,0xe9,0x01,0x04,0x82,0x03]

tbuffer_load_format_xyz v[4:6], v[1:2], s[8:11], s0, format:34 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0x11,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_UINT] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0x11,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3, format:[BUF_FMT_10_10_10_2_SINT] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0x19,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3, format:35 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0x19,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0x19,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xyzw v[4:7], off, s[8:11], s3, format:[BUF_FMT_2_10_10_10_UNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x21,0xe9,0x00,0x04,0x02,0x03]

tbuffer_load_format_xyzw v[252:255], off, s[8:11], s3, format:36 offset:4095
// GFX11: encoding: [0xff,0x8f,0x21,0xe9,0x00,0xfc,0x02,0x03]

tbuffer_load_format_xyzw v[4:7], off, s[12:15], s3, format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x21,0xe9,0x00,0x04,0x03,0x03]

tbuffer_load_format_xyzw v[4:7], off, s[12:15], s101, format:[BUF_FMT_2_10_10_10_SNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x29,0xe9,0x00,0x04,0x03,0x65]

tbuffer_load_format_xyzw v[4:7], off, s[12:15], m0, format:37 offset:4095
// GFX11: encoding: [0xff,0x8f,0x29,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_load_format_xyzw v[4:7], off, s[8:11], 0, format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_SNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x29,0xe9,0x00,0x04,0x02,0x80]

tbuffer_load_format_xyzw v[4:7], off, s[8:11], 61, format:[BUF_FMT_2_10_10_10_USCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x31,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], 61, format:38 offset:4095
// GFX11: encoding: [0xff,0x8f,0x31,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_load_format_xyzw v[4:7], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX11: encoding: [0x34,0x80,0x31,0xe9,0x01,0x04,0x42,0x03]

tbuffer_load_format_xyzw v[4:7], v1, s[8:11], s3, format:[BUF_FMT_2_10_10_10_SSCALED] idxen offset:52
// GFX11: encoding: [0x34,0x80,0x39,0xe9,0x01,0x04,0x82,0x03]

tbuffer_load_format_xyzw v[4:7], v[1:2], s[8:11], s0, format:39 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0x39,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_SSCALED] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0x39,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3, format:[BUF_FMT_2_10_10_10_UINT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0x41,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3, format:40 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0x41,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0x41,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_x v4, off, s[8:11], s3, format:[BUF_FMT_2_10_10_10_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x4e,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_d16_format_x v255, off, s[8:11], s3, format:41 offset:4095
// GFX11: encoding: [0xff,0x0f,0x4e,0xe9,0x00,0xff,0x02,0x03]

tbuffer_store_d16_format_x v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x4e,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_d16_format_x v4, off, s[12:15], s101, format:[BUF_FMT_8_8_8_8_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x56,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_d16_format_x v4, off, s[12:15], m0, format:42 offset:4095
// GFX11: encoding: [0xff,0x0f,0x56,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_d16_format_x v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x56,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_d16_format_x v4, off, s[8:11], 61, format:[BUF_FMT_8_8_8_8_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x5e,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_d16_format_x v4, off, ttmp[4:7], 61, format:43 offset:4095
// GFX11: encoding: [0xff,0x0f,0x5e,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_d16_format_x v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX11: encoding: [0x34,0x00,0x5e,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_d16_format_x v4, v1, s[8:11], s3, format:[BUF_FMT_8_8_8_8_USCALED] idxen offset:52
// GFX11: encoding: [0x34,0x00,0x66,0xe9,0x01,0x04,0x82,0x03]

tbuffer_store_d16_format_x v4, v[1:2], s[8:11], s0, format:44 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0x66,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_USCALED] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0x66,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3, format:[BUF_FMT_8_8_8_8_SSCALED] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0x6e,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3, format:45 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0x6e,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0x6e,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xy v4, off, s[8:11], s3, format:[BUF_FMT_8_8_8_8_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x76,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_d16_format_xy v255, off, s[8:11], s3, format:46 offset:4095
// GFX11: encoding: [0xff,0x8f,0x76,0xe9,0x00,0xff,0x02,0x03]

tbuffer_store_d16_format_xy v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x76,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_d16_format_xy v4, off, s[12:15], s101, format:[BUF_FMT_8_8_8_8_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x7e,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_d16_format_xy v4, off, s[12:15], m0, format:47 offset:4095
// GFX11: encoding: [0xff,0x8f,0x7e,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_d16_format_xy v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x7e,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_d16_format_xy v4, off, s[8:11], 61, format:[BUF_FMT_32_32_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x86,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], 61, format:48 offset:4095
// GFX11: encoding: [0xff,0x8f,0x86,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_d16_format_xy v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX11: encoding: [0x34,0x80,0x86,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_d16_format_xy v4, v1, s[8:11], s3, format:[BUF_FMT_32_32_SINT] idxen offset:52
// GFX11: encoding: [0x34,0x80,0x8e,0xe9,0x01,0x04,0x82,0x03]

tbuffer_store_d16_format_xy v4, v[1:2], s[8:11], s0, format:49 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0x8e,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_SINT] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0x8e,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3, format:[BUF_FMT_32_32_FLOAT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0x96,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3, format:50 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0x96,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0x96,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_16_16_16_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x9f,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_d16_format_xyz v[254:255], off, s[8:11], s3, format:51 offset:4095
// GFX11: encoding: [0xff,0x0f,0x9f,0xe9,0x00,0xfe,0x02,0x03]

tbuffer_store_d16_format_xyz v[4:5], off, s[12:15], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x9f,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_d16_format_xyz v[4:5], off, s[12:15], s101, format:[BUF_FMT_16_16_16_16_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0xa7,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_d16_format_xyz v[4:5], off, s[12:15], m0, format:52 offset:4095
// GFX11: encoding: [0xff,0x0f,0xa7,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], 0, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0xa7,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], 61, format:[BUF_FMT_16_16_16_16_USCALED] offset:4095
// GFX11: encoding: [0xff,0x0f,0xaf,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], 61, format:53 offset:4095
// GFX11: encoding: [0xff,0x0f,0xaf,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_d16_format_xyz v[4:5], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX11: encoding: [0x34,0x00,0xaf,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_d16_format_xyz v[4:5], v1, s[8:11], s3, format:[BUF_FMT_16_16_16_16_SSCALED] idxen offset:52
// GFX11: encoding: [0x34,0x00,0xb7,0xe9,0x01,0x04,0x82,0x03]

tbuffer_store_d16_format_xyz v[4:5], v[1:2], s[8:11], s0, format:54 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0xb7,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SSCALED] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0xb7,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_16_16_16_16_UINT] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0xbf,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3, format:55 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0xbf,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0xbf,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_16_16_16_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xc7,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_d16_format_xyzw v[254:255], off, s[8:11], s3, format:56 offset:4095
// GFX11: encoding: [0xff,0x8f,0xc7,0xe9,0x00,0xfe,0x02,0x03]

tbuffer_store_d16_format_xyzw v[4:5], off, s[12:15], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xc7,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_d16_format_xyzw v[4:5], off, s[12:15], s101, format:[BUF_FMT_16_16_16_16_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xcf,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_d16_format_xyzw v[4:5], off, s[12:15], m0, format:57 offset:4095
// GFX11: encoding: [0xff,0x8f,0xcf,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], 0, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xcf,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], 61, format:[BUF_FMT_32_32_32_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xd7,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], 61, format:58 offset:4095
// GFX11: encoding: [0xff,0x8f,0xd7,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_d16_format_xyzw v[4:5], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX11: encoding: [0x34,0x80,0xd7,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_d16_format_xyzw v[4:5], v1, s[8:11], s3, format:[BUF_FMT_32_32_32_SINT] idxen offset:52
// GFX11: encoding: [0x34,0x80,0xdf,0xe9,0x01,0x04,0x82,0x03]

tbuffer_store_d16_format_xyzw v[4:5], v[1:2], s[8:11], s0, format:59 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0xdf,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_SINT] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0xdf,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_32_32_32_FLOAT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0xe7,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3, format:60 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0xe7,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0xe7,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_x v4, off, s[8:11], s3, format:[BUF_FMT_32_32_32_32_UINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xea,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_format_x v255, off, s[8:11], s3, format:61 offset:4095
// GFX11: encoding: [0xff,0x0f,0xea,0xe9,0x00,0xff,0x02,0x03]

tbuffer_store_format_x v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_32_32_32_32, BUF_NUM_FORMAT_UINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xea,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_format_x v4, off, s[12:15], s101, format:[BUF_FMT_32_32_32_32_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xf2,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_format_x v4, off, s[12:15], m0, format:62 offset:4095
// GFX11: encoding: [0xff,0x0f,0xf2,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_format_x v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_32_32_32_32, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xf2,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_format_x v4, off, s[8:11], 61, format:[BUF_FMT_32_32_32_32_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xfa,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_format_x v4, off, ttmp[4:7], 61, format:63 offset:4095
// GFX11: encoding: [0xff,0x0f,0xfa,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_format_x v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_32_32_32_32, BUF_NUM_FORMAT_FLOAT] offen offset:52
// GFX11: encoding: [0x34,0x00,0xfa,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_format_x v4, v1, s[8:11], s3, format:[BUF_FMT_8_UNORM] idxen offset:52
// GFX11: encoding: [0x34,0x00,0x0a,0xe8,0x01,0x04,0x82,0x03]

tbuffer_store_format_x v4, v[1:2], s[8:11], s0, format:[BUF_FMT_8_SNORM] idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0x12,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_store_format_x v4, off, ttmp[4:7], s3, format:[BUF_FMT_8_USCALED] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0x1a,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_store_format_x v4, off, ttmp[4:7], s3, format:[BUF_FMT_8_SSCALED] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0x22,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_store_format_x v4, off, ttmp[4:7], s3, format:[BUF_FMT_8_UINT] offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0x2a,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_store_format_x v4, off, ttmp[4:7], s3, format:[BUF_FMT_8_SINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0x32,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xy v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_UNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x3a,0xe8,0x00,0x04,0x02,0x03]

tbuffer_store_format_xy v[254:255], off, s[8:11], s3, format:[BUF_FMT_16_SNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x42,0xe8,0x00,0xfe,0x02,0x03]

tbuffer_store_format_xy v[4:5], off, s[12:15], s3, format:[BUF_FMT_16_USCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x4a,0xe8,0x00,0x04,0x03,0x03]

tbuffer_store_format_xy v[4:5], off, s[12:15], s101, format:[BUF_FMT_16_SSCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x52,0xe8,0x00,0x04,0x03,0x65]

tbuffer_store_format_xy v[4:5], off, s[12:15], m0, format:[BUF_FMT_16_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x5a,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_store_format_xy v[4:5], off, s[8:11], 0, format:[BUF_FMT_16_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x62,0xe8,0x00,0x04,0x02,0x80]

tbuffer_store_format_xy v[4:5], off, s[8:11], 61, format:[BUF_FMT_16_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x6a,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], 61, format:[BUF_FMT_8_8_UNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x72,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_store_format_xy v[4:5], v1, s[8:11], s3, format:[BUF_FMT_8_8_SNORM] offen offset:52
// GFX11: encoding: [0x34,0x80,0x7a,0xe8,0x01,0x04,0x42,0x03]

tbuffer_store_format_xy v[4:5], v1, s[8:11], s3, format:[BUF_FMT_8_8_USCALED] idxen offset:52
// GFX11: encoding: [0x34,0x80,0x82,0xe8,0x01,0x04,0x82,0x03]

tbuffer_store_format_xy v[4:5], v[1:2], s[8:11], s0, format:[BUF_FMT_8_8_SSCALED] idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0x8a,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_8_8_UINT] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0x92,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_8_8_SINT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0x9a,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_32_UINT] offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0xa2,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_32_SINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0xaa,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xyz v[4:6], off, s[8:11], s3, format:[BUF_FMT_32_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xb3,0xe8,0x00,0x04,0x02,0x03]

tbuffer_store_format_xyz v[253:255], off, s[8:11], s3, format:[BUF_FMT_16_16_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0xbb,0xe8,0x00,0xfd,0x02,0x03]

tbuffer_store_format_xyz v[4:6], off, s[12:15], s3, format:[BUF_FMT_16_16_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0xc3,0xe8,0x00,0x04,0x03,0x03]

tbuffer_store_format_xyz v[4:6], off, s[12:15], s101, format:[BUF_FMT_16_16_USCALED] offset:4095
// GFX11: encoding: [0xff,0x0f,0xcb,0xe8,0x00,0x04,0x03,0x65]

tbuffer_store_format_xyz v[4:6], off, s[12:15], m0, format:[BUF_FMT_16_16_SSCALED] offset:4095
// GFX11: encoding: [0xff,0x0f,0xd3,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_store_format_xyz v[4:6], off, s[8:11], 0, format:[BUF_FMT_16_16_UINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xdb,0xe8,0x00,0x04,0x02,0x80]

tbuffer_store_format_xyz v[4:6], off, s[8:11], 61, format:[BUF_FMT_16_16_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xe3,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], 61, format:[BUF_FMT_16_16_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x0f,0xeb,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_store_format_xyz v[4:6], v1, s[8:11], s3, format:[BUF_FMT_10_11_11_FLOAT] offen offset:52
// GFX11: encoding: [0x34,0x00,0xf3,0xe8,0x01,0x04,0x42,0x03]

tbuffer_store_format_xyz v[4:6], v1, s[8:11], s3, format:[BUF_FMT_11_11_10_FLOAT] idxen offset:52
// GFX11: encoding: [0x34,0x00,0xfb,0xe8,0x01,0x04,0x82,0x03]

tbuffer_store_format_xyz v[4:6], v[1:2], s[8:11], s0, format:[BUF_FMT_10_10_10_2_UNORM] idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0x03,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3, format:[BUF_FMT_10_10_10_2_SNORM] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0x0b,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3, format:[BUF_FMT_10_10_10_2_UINT] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0x13,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3, format:[BUF_FMT_10_10_10_2_SINT]  offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0x1b,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3, format:[BUF_FMT_2_10_10_10_UNORM] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0x23,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xyzw v[4:7], off, s[8:11], s3, format:[BUF_FMT_2_10_10_10_SNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x2b,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_format_xyzw v[252:255], off, s[8:11], s3, format:[BUF_FMT_2_10_10_10_USCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x33,0xe9,0x00,0xfc,0x02,0x03]

tbuffer_store_format_xyzw v[4:7], off, s[12:15], s3, format:[BUF_FMT_2_10_10_10_SSCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x3b,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_format_xyzw v[4:7], off, s[12:15], s101, format:[BUF_FMT_2_10_10_10_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x43,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_format_xyzw v[4:7], off, s[12:15], m0, format:[BUF_FMT_2_10_10_10_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x4b,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_format_xyzw v[4:7], off, s[8:11], 0, format:[BUF_FMT_8_8_8_8_UNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x53,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_format_xyzw v[4:7], off, s[8:11], 61, format:[BUF_FMT_8_8_8_8_SNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x5b,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], 61, format:[BUF_FMT_8_8_8_8_USCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x63,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_format_xyzw v[4:7], v1, s[8:11], s3, format:[BUF_FMT_8_8_8_8_SSCALED] offen offset:52
// GFX11: encoding: [0x34,0x80,0x6b,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_format_xyzw v[4:7], v1, s[8:11], s3, format:[BUF_FMT_8_8_8_8_UINT] idxen offset:52
// GFX11: encoding: [0x34,0x80,0x73,0xe9,0x01,0x04,0x82,0x03]

tbuffer_store_format_xyzw v[4:7], v[1:2], s[8:11], s0, format:[BUF_FMT_8_8_8_8_SINT] idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0x7b,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3, format:[BUF_FMT_32_32_UINT] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0x83,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3, format:[BUF_FMT_32_32_SINT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0x8b,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3, format:[BUF_FMT_32_32_FLOAT] offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0x93,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3, format:[BUF_FMT_16_16_16_16_UNORM] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0x9b,0xe9,0x00,0x04,0x1c,0x03]

//Removed formats (compared to gfx10)

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_UNORM] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_SNORM] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_USCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_SSCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_UINT] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_SINT] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_UNORM] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_SNORM] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_USCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_SSCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_UINT] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_10_10_10_2_USCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3, format:[BUF_FMT_10_10_10_2_SSCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format
