// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefixes=GFX1250 %s

ds_load_tr_b64 v[2:3], v0
// GFX1250: ds_load_tr8_b64 v[2:3], v0              ; encoding: [0x00,0x00,0xf4,0xdb,0x00,0x00,0x00,0x02]

ds_load_tr_b128 v[2:5], v0
// GFX1250: ds_load_tr16_b128 v[2:5], v0            ; encoding: [0x00,0x00,0xf0,0xdb,0x00,0x00,0x00,0x02]
