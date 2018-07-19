// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 -check-prefix=GCN %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding 2>&1 %s | FileCheck -check-prefix=VI-ERR -check-prefix=GCNERR %s

buffer_load_ubyte_d16 v1, off, s[4:7], s1
// VI-ERR: error: instruction not supported on this GPU
// GFX9: buffer_load_ubyte_d16 v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x80,0xe0,0x00,0x01,0x01,0x01]

buffer_load_ubyte_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_load_ubyte_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x84,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: error: instruction not supported on this GPU

buffer_load_sbyte_d16 v1, off, s[4:7], s1
// GFX9: buffer_load_sbyte_d16 v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x88,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: error: instruction not supported on this GPU

buffer_load_sbyte_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_load_sbyte_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x8c,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: error: instruction not supported on this GPU

buffer_load_short_d16 v1, off, s[4:7], s1
// GFX9: buffer_load_short_d16 v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x90,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: error: instruction not supported on this GPU

buffer_load_short_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_load_short_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x94,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: error: instruction not supported on this GPU

buffer_store_byte_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_store_byte_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x64,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: error: instruction not supported on this GPU

buffer_store_short_d16_hi v1, off, s[4:7], s1
// GFX9: buffer_store_short_d16_hi v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x6c,0xe0,0x00,0x01,0x01,0x01]
// VI-ERR: error: instruction not supported on this GPU
