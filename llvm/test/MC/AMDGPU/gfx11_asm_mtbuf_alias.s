// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s

tbuffer_load_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_8_UNORM] offset:4095
// GFX11: tbuffer_load_d16_format_x v4, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_d16_xy v4, off, s[8:11], s3, format:[BUF_FMT_8_SINT] offset:4095
// GFX11: tbuffer_load_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_SINT] offset:4095 ; encoding: [0xff,0x8f,0x34,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_d16_xyz v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_UINT] offset:4095
// GFX11: tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UINT] offset:4095 ; encoding: [0xff,0x0f,0x5d,0xe8,0x00,0x04,0x02,0x03]

tbuffer_load_format_d16_xyzw v[4:5], off, s[8:11], s3, format:[BUF_FMT_8_8_USCALED] offset:4095
// GFX11: tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_8_8_USCALED] offset:4095 ; encoding: [0xff,0x8f,0x85,0xe8,0x00,0x04,0x02,0x03]

tbuffer_store_format_d16_x v4, off, s[8:11], s3, format:[BUF_FMT_2_10_10_10_SINT] offset:4095
// GFX11: tbuffer_store_d16_format_x v4, off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SINT] offset:4095 ; encoding: [0xff,0x0f,0x4e,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_format_d16_xy v4, off, s[8:11], s3, format:[BUF_FMT_8_8_8_8_UINT] offset:4095
// GFX11: tbuffer_store_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_8_8_8_UINT] offset:4095 ; encoding: [0xff,0x8f,0x76,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_format_d16_xyz v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_16_16_16_UNORM] offset:4095
// GFX11: tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:4095 ; encoding: [0xff,0x0f,0x9f,0xe9,0x00,0x04,0x02,0x03]

tbuffer_store_format_d16_xyzw v[4:5], off, s[8:11], s3, format:[BUF_FMT_16_16_16_16_SINT] offset:4095
// GFX11: tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SINT] offset:4095 ; encoding: [0xff,0x8f,0xc7,0xe9,0x00,0x04,0x02,0x03]
