// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s | FileCheck --check-prefix=GFX12 %s

tbuffer_load_format_d16_x v4, off, s[8:11], s3 format:[BUF_FMT_8_UNORM] offset:8388607
// GFX12: tbuffer_load_d16_format_x v4, off, s[8:11], s3 offset:8388607 ; encoding: [0x03,0x00,0x22,0xc4,0x04,0x10,0x80,0x00,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_SINT] offset:8388607
// GFX12: tbuffer_load_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_SINT] offset:8388607 ; encoding: [0x03,0x40,0x22,0xc4,0x04,0x10,0x00,0x03,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UINT] offset:8388607
// GFX12: tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UINT] offset:8388607 ; encoding: [0x03,0x80,0x22,0xc4,0x04,0x10,0x80,0x05,0x00,0xff,0xff,0x7f]

tbuffer_load_format_d16_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_8_8_USCALED] offset:8388607
// GFX12: tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_8_8_USCALED] offset:8388607 ; encoding: [0x03,0xc0,0x22,0xc4,0x04,0x10,0x00,0x08,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_x v4, off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SINT] offset:8388607
// GFX12: tbuffer_store_d16_format_x v4, off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SINT] offset:8388607 ; encoding: [0x03,0x00,0x23,0xc4,0x04,0x10,0x80,0x14,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_8_8_8_UINT] offset:8388607
// GFX12: tbuffer_store_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_8_8_8_UINT] offset:8388607 ; encoding: [0x03,0x40,0x23,0xc4,0x04,0x10,0x00,0x17,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607
// GFX12: tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607 ; encoding: [0x03,0x80,0x23,0xc4,0x04,0x10,0x80,0x19,0x00,0xff,0xff,0x7f]

tbuffer_store_format_d16_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SINT] offset:8388607
// GFX12: tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SINT] offset:8388607 ; encoding: [0x03,0xc0,0x23,0xc4,0x04,0x10,0x00,0x1c,0x00,0xff,0xff,0x7f]
