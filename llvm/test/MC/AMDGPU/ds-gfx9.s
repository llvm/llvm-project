// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 -check-prefix=GCN %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding 2>&1 %s | FileCheck -check-prefix=VI-ERR -check-prefix=GCNERR %s

ds_read_u8_d16 v8, v2
// GFX9: ds_read_u8_d16 v8, v2           ; encoding: [0x00,0x00,0xac,0xd8,0x02,0x00,0x00,0x08]
// VI-ERR: error: instruction not supported on this GPU

ds_read_u8_d16_hi v8, v2
// GFX9: ds_read_u8_d16_hi v8, v2        ; encoding: [0x00,0x00,0xae,0xd8,0x02,0x00,0x00,0x08]
// VI-ERR: error: instruction not supported on this GPU

ds_read_i8_d16 v8, v2
// GFX9: ds_read_i8_d16 v8, v2           ; encoding: [0x00,0x00,0xb0,0xd8,0x02,0x00,0x00,0x08]
// VI-ERR: error: instruction not supported on this GPU

ds_read_i8_d16_hi v8, v2
// GFX9: ds_read_i8_d16_hi v8, v2        ; encoding: [0x00,0x00,0xb2,0xd8,0x02,0x00,0x00,0x08]
// VI-ERR: error: instruction not supported on this GPU

ds_read_u16_d16 v8, v2
// GFX9: ds_read_u16_d16 v8, v2          ; encoding: [0x00,0x00,0xb4,0xd8,0x02,0x00,0x00,0x08]
// VI-ERR: error: instruction not supported on this GPU

ds_read_u16_d16_hi v8, v2
// GFX9: ds_read_u16_d16_hi v8, v2       ; encoding: [0x00,0x00,0xb6,0xd8,0x02,0x00,0x00,0x08]
// VI-ERR: error: instruction not supported on this GPU

ds_write_b8_d16_hi v8, v2
// VI-ERR: error: instruction not supported on this GPU
// GFX9: ds_write_b8_d16_hi v8, v2       ; encoding: [0x00,0x00,0xa8,0xd8,0x08,0x02,0x00,0x00]

ds_write_b16_d16_hi v8, v2
// VI-ERR: error: instruction not supported on this GPU
// GFX9: ds_write_b16_d16_hi v8, v2      ; encoding: [0x00,0x00,0xaa,0xd8,0x08,0x02,0x00,0x00]

ds_write_addtid_b32 v8, v2
// VI-ERR: error: instruction not supported on this GPU
// GFX9: ds_write_addtid_b32 v8, v2      ; encoding: [0x00,0x00,0x3a,0xd8,0x08,0x02,0x00,0x00]

ds_read_addtid_b32 v8, v2
// VI-ERR: error: instruction not supported on this GPU
// GFX9: ds_read_addtid_b32 v8, v2       ; encoding: [0x00,0x00,0x6c,0xd9,0x02,0x00,0x00,0x08]
