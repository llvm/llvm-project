// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=GFX10 %s

// GFX10: tbuffer_load_format_d16_x v0, off, s[0:3], format:22, 0 ; encoding: [0x00,0x00,0xb0,0xe8,0x00,0x00,0x20,0x80]
tbuffer_load_format_d16_x v0, off, s[0:3], format:22, 0
// GFX10: tbuffer_load_format_d16_xy v0, off, s[0:3], format:22, 0 ; encoding: [0x00,0x00,0xb1,0xe8,0x00,0x00,0x20,0x80]
tbuffer_load_format_d16_xy v0, off, s[0:3], format:22, 0
// GFX10: tbuffer_load_format_d16_xyzw v[0:1], off, s[0:3], format:22, 0 ; encoding: [0x00,0x00,0xb3,0xe8,0x00,0x00,0x20,0x80]
tbuffer_load_format_d16_xyzw v[0:1], off, s[0:3], format:22, 0
// GFX10: tbuffer_load_format_xyzw v[0:3], off, s[0:3], format:78, 0 ; encoding: [0x00,0x00,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], off, s[0:3], format:78, 0
// GFX10: tbuffer_load_format_xyzw v[8:11], off, s[0:3], format:22, 0 slc ; encoding: [0x00,0x00,0xb3,0xe8,0x00,0x08,0x40,0x80]
tbuffer_load_format_xyzw v[8:11], off, s[0:3], format:22, 0 slc
// GFX10: tbuffer_load_format_xyzw v[4:7], off, s[0:3], format:63, 0 glc ; encoding: [0x00,0x40,0xfb,0xe9,0x00,0x04,0x00,0x80]
tbuffer_load_format_xyzw v[4:7], off, s[0:3], format:63, 0 glc
// GFX10: tbuffer_load_format_xyzw v[12:15], off, s[0:3], format:23, 0 glc dlc ; encoding: [0x00,0xc0,0xbb,0xe8,0x00,0x0c,0x00,0x80]
tbuffer_load_format_xyzw v[12:15], off, s[0:3], format:23, 0 glc dlc
// GFX10: tbuffer_load_format_xyzw v[0:3], off, s[0:3], format:78, 0 offset:42 ; encoding: [0x2a,0x00,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], off, s[0:3], format:78, 0 offset:42
// GFX10: tbuffer_load_format_xyzw v[4:7], off, s[0:3], format:62, s4 offset:73 ; encoding: [0x49,0x00,0xf3,0xe9,0x00,0x04,0x00,0x04]
tbuffer_load_format_xyzw v[4:7], off, s[0:3], format:62, s4 offset:73
// GFX10: tbuffer_load_format_xyzw v[0:3], off, s[0:3], format:47, 61 offset:4095 ; encoding: [0xff,0x0f,0x7b,0xe9,0x00,0x00,0x00,0xbd]
tbuffer_load_format_xyzw v[0:3], off, s[0:3], format:47, 61 offset:4095
// GFX10: tbuffer_load_format_xyzw v[8:11], off, s[0:3], format:77, s4 offset:1 ; encoding: [0x01,0x00,0x6b,0xea,0x00,0x08,0x00,0x04]
tbuffer_load_format_xyzw v[8:11], off, s[0:3], format:77, s4 offset:1
// GFX10: tbuffer_load_format_xyzw v[0:3], v0, s[0:3], format:78, 0 idxen ; encoding: [0x00,0x20,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], v0, s[0:3], format:78, 0 idxen
// GFX10: tbuffer_load_format_xyzw v[0:3], v0, s[0:3], format:78, 0 offen ; encoding: [0x00,0x10,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], v0, s[0:3], format:78, 0 offen
// GFX10: tbuffer_load_format_xyzw v[0:3], v0, s[0:3], format:78, 0 offen offset:52 ; encoding: [0x34,0x10,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], v0, s[0:3], format:78, 0 offen offset:52
// GFX10: tbuffer_load_format_xyzw v[0:3], v[0:1], s[0:3], format:78, 0 idxen offen ; encoding: [0x00,0x30,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], v[0:1], s[0:3], format:78, 0 idxen offen
// GFX10: tbuffer_load_format_xy v[0:1], off, s[0:3], format:77, 0 ; encoding: [0x00,0x00,0x69,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xy v[0:1], off, s[0:3], format:77, 0
// GFX10: tbuffer_load_format_x v0, off, s[0:3], format:77, 0 ; encoding: [0x00,0x00,0x68,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_x v0, off, s[0:3], format:77, 0
// GFX10: tbuffer_store_format_d16_x v0, v1, s[4:7], format:33, 0 idxen ; encoding: [0x00,0x20,0x0c,0xe9,0x01,0x00,0x21,0x80]
tbuffer_store_format_d16_x v0, v1, s[4:7], format:33, 0 idxen
// GFX10: tbuffer_store_format_d16_xy v0, v1, s[4:7], format:33, 0 idxen ; encoding: [0x00,0x20,0x0d,0xe9,0x01,0x00,0x21,0x80]
tbuffer_store_format_d16_xy v0, v1, s[4:7], format:33, 0 idxen
// GFX10: tbuffer_store_format_d16_xyzw v[0:1], v2, s[4:7], format:33, 0 idxen ; encoding: [0x00,0x20,0x0f,0xe9,0x02,0x00,0x21,0x80]
tbuffer_store_format_d16_xyzw v[0:1], v2, s[4:7], format:33, 0 idxen
// GFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], format:44, 0 ; encoding: [0x00,0x00,0x67,0xe9,0x00,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], off, s[0:3], format:44, 0
// GFX10: tbuffer_store_format_xyzw v[4:7], off, s[0:3], format:61, 0 glc ; encoding: [0x00,0x40,0xef,0xe9,0x00,0x04,0x00,0x80]
tbuffer_store_format_xyzw v[4:7], off, s[0:3], format:61, 0 glc
// GFX10: tbuffer_store_format_xyzw v[8:11], off, s[0:3], format:78, 0 slc ; encoding: [0x00,0x00,0x77,0xea,0x00,0x08,0x40,0x80]
tbuffer_store_format_xyzw v[8:11], off, s[0:3], format:78, 0 slc
// GFX10: tbuffer_store_format_xyzw v[8:11], off, s[0:3], format:78, 0 ; encoding: [0x00,0x00,0x77,0xea,0x00,0x08,0x00,0x80]
tbuffer_store_format_xyzw v[8:11], off, s[0:3], format:78, 0
// GFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], format:117, 0 offset:42 ; encoding: [0x2a,0x00,0xaf,0xeb,0x00,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], off, s[0:3], format:117, 0 offset:42
// GFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], format:117, s4 offset:42 ; encoding: [0x2a,0x00,0xaf,0xeb,0x00,0x00,0x00,0x04]
tbuffer_store_format_xyzw v[0:3], off, s[0:3], format:117, s4 offset:42
// GFX10: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], format:47, 0 idxen ; encoding: [0x00,0x20,0x7f,0xe9,0x04,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v4, s[0:3], format:47, 0 idxen
// GFX10: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], format:115, 0 offen ; encoding: [0x00,0x10,0x9f,0xeb,0x04,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v4, s[0:3], format:115, 0 offen
// GFX10: tbuffer_store_format_xyzw v[0:3], v[4:5], s[0:3], format:70, 0 idxen offen ; encoding: [0x00,0x30,0x37,0xea,0x04,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v[4:5], s[0:3], format:70, 0 idxen offen
// GFX10: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], format:63, 0 idxen ; encoding: [0x00,0x20,0xff,0xe9,0x04,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v4, s[0:3], format:63, 0 idxen
// GFX10: tbuffer_store_format_xyzw v[0:3], v6, s[0:3], format:46, 0 idxen ; encoding: [0x00,0x20,0x77,0xe9,0x06,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v6, s[0:3], format:46, 0 idxen
// GFX10: tbuffer_store_format_x v0, v1, s[0:3], format:125, 0 idxen ; encoding: [0x00,0x20,0xec,0xeb,0x01,0x00,0x00,0x80]
tbuffer_store_format_x v0, v1, s[0:3], format:125, 0 idxen
// GFX10: tbuffer_store_format_xy v[0:1], v2, s[0:3], format:33, 0 idxen ; encoding: [0x00,0x20,0x0d,0xe9,0x02,0x00,0x00,0x80]
tbuffer_store_format_xy v[0:1], v2, s[0:3], format:33, 0 idxen
