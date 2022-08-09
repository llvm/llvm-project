// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX11-ERR --implicit-check-not=error: %s

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_8_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x0c,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_d16_x v255, off, s[8:11], s3, format:1 offset:4095
// GFX11: encoding: [0xff,0x0f,0x0c,0xe8,0x00,0xff,0x02,0x03]

tbuffer_load_format_d16_x v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x0c,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_format_d16_x v4, off, s[12:15], s101, format:[BUF_FMT_8_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x14,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_format_d16_x v4, off, s[12:15], m0, format:2 offset:4095
// GFX11: encoding: [0xff,0x0f,0x14,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_format_d16_x v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x14,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_format_d16_x v4, off, s[8:11], 61, format:[BUF_FMT_8_USCALED] offset:4095
// GFX11: encoding: [0xff,0x0f,0x1c,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], 61, format:3 offset:4095
// GFX11: encoding: [0xff,0x0f,0x1c,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_format_d16_x v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX11: encoding: [0x34,0x00,0x1c,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_format_d16_x v4, v1, s[8:11], s3, format:[BUF_FMT_8_SSCALED] idxen offset:52
// GFX11: encoding: [0x34,0x00,0x24,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_format_d16_x v4, v[1:2], s[8:11], s0, format:4 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0x24,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SSCALED] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0x24,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3, format:[BUF_FMT_8_UINT] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0x2c,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3, format:5 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0x2c,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0x2c,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xy v4, off, s[8:11], s3, format:[BUF_FMT_8_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x34,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_d16_xy v255, off, s[8:11], s3, format:6 offset:4095
// GFX11: encoding: [0xff,0x8f,0x34,0xe8,0x00,0xff,0x02,0x03]

tbuffer_load_format_d16_xy v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x34,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_format_d16_xy v4, off, s[12:15], s101, format:[BUF_FMT_16_UNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x3c,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_format_d16_xy v4, off, s[12:15], m0, format:7 offset:4095
// GFX11: encoding: [0xff,0x8f,0x3c,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_format_d16_xy v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x3c,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_format_d16_xy v4, off, s[8:11], 61, format:[BUF_FMT_16_SNORM] offset:4095
// GFX11: encoding: [0xff,0x8f,0x44,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], 61, format:8 offset:4095
// GFX11: encoding: [0xff,0x8f,0x44,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_format_d16_xy v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX11: encoding: [0x34,0x80,0x44,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_format_d16_xy v4, v1, s[8:11], s3, format:[BUF_FMT_16_USCALED] idxen offset:52
// GFX11: encoding: [0x34,0x80,0x4c,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_format_d16_xy v4, v[1:2], s[8:11], s0, format:9 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0x4c,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_USCALED] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0x4c,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3, format:[BUF_FMT_16_SSCALED] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0x54,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3, format:10 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0x54,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xy v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0x54,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xyz v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_UINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x5d,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_d16_xyz v[254:255], off, s[8:11], s3, format:11 offset:4095
// GFX11: encoding: [0xff,0x0f,0x5d,0xe8,0x00,0xfe,0x02,0x03]

tbuffer_load_format_d16_xyz v[4:5], off, s[12:15], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_UINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x5d,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_format_d16_xyz v[4:5], off, s[12:15], s101, format:[BUF_FMT_16_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x65,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_format_d16_xyz v[4:5], off, s[12:15], m0, format:12 offset:4095
// GFX11: encoding: [0xff,0x0f,0x65,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_format_d16_xyz v[4:5], off, s[8:11], 0, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x65,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_format_d16_xyz v[4:5], off, s[8:11], 61, format:[BUF_FMT_16_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x6d,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], 61, format:13 offset:4095
// GFX11: encoding: [0xff,0x0f,0x6d,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_format_d16_xyz v[4:5], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_FLOAT] offen offset:52
// GFX11: encoding: [0x34,0x00,0x6d,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_format_d16_xyz v[4:5], v1, s[8:11], s3, format:[BUF_FMT_8_8_UNORM] idxen offset:52
// GFX11: encoding: [0x34,0x00,0x75,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_format_d16_xyz v[4:5], v[1:2], s[8:11], s0, format:14 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0x75,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_UNORM] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0x75,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_8_8_SNORM] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0x7d,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3, format:15 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0x7d,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0x7d,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xyzw v[4:5], off, s[8:11], s3, format:[BUF_FMT_8_8_USCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x85,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_d16_xyzw v[254:255], off, s[8:11], s3, format:16 offset:4095
// GFX11: encoding: [0xff,0x8f,0x85,0xe8,0x00,0xfe,0x02,0x03]

tbuffer_load_format_d16_xyzw v[4:5], off, s[12:15], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_USCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x85,0xe8,0x00,0x04,0x03,0x03]

tbuffer_load_format_d16_xyzw v[4:5], off, s[12:15], s101, format:[BUF_FMT_8_8_SSCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x8d,0xe8,0x00,0x04,0x03,0x65]

tbuffer_load_format_d16_xyzw v[4:5], off, s[12:15], m0, format:17 offset:4095
// GFX11: encoding: [0xff,0x8f,0x8d,0xe8,0x00,0x04,0x03,0x7d]

tbuffer_load_format_d16_xyzw v[4:5], off, s[8:11], 0, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SSCALED] offset:4095
// GFX11: encoding: [0xff,0x8f,0x8d,0xe8,0x00,0x04,0x02,0x80]

tbuffer_load_format_d16_xyzw v[4:5], off, s[8:11], 61, format:[BUF_FMT_8_8_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x95,0xe8,0x00,0x04,0x02,0xbd]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], 61, format:18 offset:4095
// GFX11: encoding: [0xff,0x8f,0x95,0xe8,0x00,0x04,0x1c,0xbd]

tbuffer_load_format_d16_xyzw v[4:5], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX11: encoding: [0x34,0x80,0x95,0xe8,0x01,0x04,0x42,0x03]

tbuffer_load_format_d16_xyzw v[4:5], v1, s[8:11], s3, format:[BUF_FMT_8_8_SINT] idxen offset:52
// GFX11: encoding: [0x34,0x80,0x9d,0xe8,0x01,0x04,0x82,0x03]

tbuffer_load_format_d16_xyzw v[4:5], v[1:2], s[8:11], s0, format:19 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0x9d,0xe8,0x01,0x04,0xc2,0x00]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SINT] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0x9d,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_32_UINT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0xa5,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3, format:20 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0xa5,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_load_format_d16_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0xa5,0xe8,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_2_10_10_10_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x4e,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_format_d16_x v255, off, s[8:11], s3, format:41 offset:4095
// GFX11: encoding: [0xff,0x0f,0x4e,0xe9,0x00,0xff,0x02,0x03]

tbuffer_store_format_d16_x v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x0f,0x4e,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_format_d16_x v4, off, s[12:15], s101, format:[BUF_FMT_8_8_8_8_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x56,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_format_d16_x v4, off, s[12:15], m0, format:42 offset:4095
// GFX11: encoding: [0xff,0x0f,0x56,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_format_d16_x v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x56,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_format_d16_x v4, off, s[8:11], 61, format:[BUF_FMT_8_8_8_8_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x5e,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], 61, format:43 offset:4095
// GFX11: encoding: [0xff,0x0f,0x5e,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_format_d16_x v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX11: encoding: [0x34,0x00,0x5e,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_format_d16_x v4, v1, s[8:11], s3, format:[BUF_FMT_8_8_8_8_USCALED] idxen offset:52
// GFX11: encoding: [0x34,0x00,0x66,0xe9,0x01,0x04,0x82,0x03]

tbuffer_store_format_d16_x v4, v[1:2], s[8:11], s0, format:44 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0x66,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_USCALED] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0x66,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3, format:[BUF_FMT_8_8_8_8_SSCALED] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0x6e,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3, format:45 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0x6e,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_x v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0x6e,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xy v4, off, s[8:11], s3, format:[BUF_FMT_8_8_8_8_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x76,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_format_d16_xy v255, off, s[8:11], s3, format:46 offset:4095
// GFX11: encoding: [0xff,0x8f,0x76,0xe9,0x00,0xff,0x02,0x03]

tbuffer_store_format_d16_xy v4, off, s[12:15], s3, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x76,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_format_d16_xy v4, off, s[12:15], s101, format:[BUF_FMT_8_8_8_8_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x7e,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_format_d16_xy v4, off, s[12:15], m0, format:47 offset:4095
// GFX11: encoding: [0xff,0x8f,0x7e,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_format_d16_xy v4, off, s[8:11], 0, format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x7e,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_format_d16_xy v4, off, s[8:11], 61, format:[BUF_FMT_32_32_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0x86,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], 61, format:48 offset:4095
// GFX11: encoding: [0xff,0x8f,0x86,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_format_d16_xy v4, v1, s[8:11], s3, format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX11: encoding: [0x34,0x80,0x86,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_format_d16_xy v4, v1, s[8:11], s3, format:[BUF_FMT_32_32_SINT] idxen offset:52
// GFX11: encoding: [0x34,0x80,0x8e,0xe9,0x01,0x04,0x82,0x03]

tbuffer_store_format_d16_xy v4, v[1:2], s[8:11], s0, format:49 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0x8e,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_SINT] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0x8e,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3, format:[BUF_FMT_32_32_FLOAT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0x96,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3, format:50 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0x96,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xy v4, off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0x96,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xyz v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_16_16_16_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x9f,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_format_d16_xyz v[254:255], off, s[8:11], s3, format:51 offset:4095
// GFX11: encoding: [0xff,0x0f,0x9f,0xe9,0x00,0xfe,0x02,0x03]

tbuffer_store_format_d16_xyz v[4:5], off, s[12:15], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0x9f,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_format_d16_xyz v[4:5], off, s[12:15], s101, format:[BUF_FMT_16_16_16_16_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0xa7,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_format_d16_xyz v[4:5], off, s[12:15], m0, format:52 offset:4095
// GFX11: encoding: [0xff,0x0f,0xa7,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_format_d16_xyz v[4:5], off, s[8:11], 0, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SNORM] offset:4095
// GFX11: encoding: [0xff,0x0f,0xa7,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_format_d16_xyz v[4:5], off, s[8:11], 61, format:[BUF_FMT_16_16_16_16_USCALED] offset:4095
// GFX11: encoding: [0xff,0x0f,0xaf,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], 61, format:53 offset:4095
// GFX11: encoding: [0xff,0x0f,0xaf,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_format_d16_xyz v[4:5], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX11: encoding: [0x34,0x00,0xaf,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_format_d16_xyz v[4:5], v1, s[8:11], s3, format:[BUF_FMT_16_16_16_16_SSCALED] idxen offset:52
// GFX11: encoding: [0x34,0x00,0xb7,0xe9,0x01,0x04,0x82,0x03]

tbuffer_store_format_d16_xyz v[4:5], v[1:2], s[8:11], s0, format:54 idxen offen offset:52
// GFX11: encoding: [0x34,0x00,0xb7,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SSCALED] offset:4095 glc
// GFX11: encoding: [0xff,0x4f,0xb7,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_16_16_16_16_UINT] offset:4095 slc
// GFX11: encoding: [0xff,0x1f,0xbf,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3, format:55 offset:4095 dlc
// GFX11: encoding: [0xff,0x2f,0xbf,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xyz v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0x7f,0xbf,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xyzw v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_16_16_16_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xc7,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_format_d16_xyzw v[254:255], off, s[8:11], s3, format:56 offset:4095
// GFX11: encoding: [0xff,0x8f,0xc7,0xe9,0x00,0xfe,0x02,0x03]

tbuffer_store_format_d16_xyzw v[4:5], off, s[12:15], s3, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xc7,0xe9,0x00,0x04,0x03,0x03]

tbuffer_store_format_d16_xyzw v[4:5], off, s[12:15], s101, format:[BUF_FMT_16_16_16_16_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xcf,0xe9,0x00,0x04,0x03,0x65]

tbuffer_store_format_d16_xyzw v[4:5], off, s[12:15], m0, format:57 offset:4095
// GFX11: encoding: [0xff,0x8f,0xcf,0xe9,0x00,0x04,0x03,0x7d]

tbuffer_store_format_d16_xyzw v[4:5], off, s[8:11], 0, format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_FLOAT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xcf,0xe9,0x00,0x04,0x02,0x80]

tbuffer_store_format_d16_xyzw v[4:5], off, s[8:11], 61, format:[BUF_FMT_32_32_32_UINT] offset:4095
// GFX11: encoding: [0xff,0x8f,0xd7,0xe9,0x00,0x04,0x02,0xbd]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], 61, format:58 offset:4095
// GFX11: encoding: [0xff,0x8f,0xd7,0xe9,0x00,0x04,0x1c,0xbd]

tbuffer_store_format_d16_xyzw v[4:5], v1, s[8:11], s3, format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX11: encoding: [0x34,0x80,0xd7,0xe9,0x01,0x04,0x42,0x03]

tbuffer_store_format_d16_xyzw v[4:5], v1, s[8:11], s3, format:[BUF_FMT_32_32_32_SINT] idxen offset:52
// GFX11: encoding: [0x34,0x80,0xdf,0xe9,0x01,0x04,0x82,0x03]

tbuffer_store_format_d16_xyzw v[4:5], v[1:2], s[8:11], s0, format:59 idxen offen offset:52
// GFX11: encoding: [0x34,0x80,0xdf,0xe9,0x01,0x04,0xc2,0x00]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_SINT] offset:4095 glc
// GFX11: encoding: [0xff,0xcf,0xdf,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_FMT_32_32_32_FLOAT] offset:4095 slc
// GFX11: encoding: [0xff,0x9f,0xe7,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3, format:60 offset:4095 dlc
// GFX11: encoding: [0xff,0xaf,0xe7,0xe9,0x00,0x04,0x1c,0x03]

tbuffer_store_format_d16_xyzw v[4:5], off, ttmp[4:7], s3, format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:4095 glc slc dlc
// GFX11: encoding: [0xff,0xff,0xe7,0xe9,0x00,0x04,0x1c,0x03]

//Removed formats (compared to gfx10)

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_UNORM] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_SNORM] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_USCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_SSCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_UINT] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_10_11_11_SINT] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_UNORM] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_SNORM] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_USCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_SSCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_11_11_10_UINT] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_10_10_10_2_USCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_10_10_10_2_SSCALED] offset:4095
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format
