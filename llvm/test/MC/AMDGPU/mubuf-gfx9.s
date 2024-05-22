// RUN: llvm-mc -triple=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga 2>&1 %s | FileCheck -check-prefix=VI-ERR --implicit-check-not=error: %s

buffer_load_ubyte_d16 v1, off, s[4:7], s1
// VI-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: buffer_load_ubyte_d16 v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x80,0xe0,0x00,0x01,0x01,0x01]

buffer_load_ubyte_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_load_ubyte_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x84,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_ubyte_d16_hi v[1:2], off, s[4:7], s1 tfe
// GFX9: buffer_load_ubyte_d16_hi v[1:2], off, s[4:7], s1 tfe ; encoding: [0x00,0x00,0x84,0xe0,0x00,0x01,0x81,0x01]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_sbyte_d16 v1, off, s[4:7], s1
// GFX9: buffer_load_sbyte_d16 v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x88,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_sbyte_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_load_sbyte_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x8c,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_short_d16 v1, off, s[4:7], s1
// GFX9: buffer_load_short_d16 v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x90,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_short_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_load_short_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x94,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_byte_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_store_byte_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x64,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_short_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_store_short_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x6c,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, off, s[8:11], s3
// GFX9: buffer_load_format_d16_hi_x v5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x98,0xe0,0x00,0x05,0x02,0x03]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:4095
// GFX9: buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe0,0x00,0x05,0x02,0x03]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, v0, s[8:11], s3 idxen offset:4095
// GFX9: buffer_load_format_d16_hi_x v5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x98,0xe0,0x00,0x05,0x02,0x03]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, v0, s[8:11], s3 offen offset:4095
// GFX9: buffer_load_format_d16_hi_x v5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x98,0xe0,0x00,0x05,0x02,0x03]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:4095 glc
// GFX9: buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x98,0xe0,0x00,0x05,0x02,0x03]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:4095 slc
// GFX9: buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x9a,0xe0,0x00,0x05,0x02,0x03]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v255, off, s[12:15], s4
// GFX9: buffer_store_format_d16_hi_x v255, off, s[12:15], s4 ; encoding: [0x00,0x00,0x9c,0xe0,0x00,0xff,0x03,0x04]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v255, off, s[12:15], s4 offset:4095
// GFX9: buffer_store_format_d16_hi_x v255, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe0,0x00,0xff,0x03,0x04]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v1, v0, s[12:15], s4 idxen offset:4095
// GFX9: buffer_store_format_d16_hi_x v1, v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x9c,0xe0,0x00,0x01,0x03,0x04]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v1, v0, s[12:15], s4 offen offset:4095
// GFX9: buffer_store_format_d16_hi_x v1, v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x9c,0xe0,0x00,0x01,0x03,0x04]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v1, off, s[12:15], s4 offset:4095 glc
// GFX9: buffer_store_format_d16_hi_x v1, off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x9c,0xe0,0x00,0x01,0x03,0x04]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v1, off, s[12:15], s4 offset:4095 slc
// GFX9: buffer_store_format_d16_hi_x v1, off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x9e,0xe0,0x00,0x01,0x03,0x04]
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
