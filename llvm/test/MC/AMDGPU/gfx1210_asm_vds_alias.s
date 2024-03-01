// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefixes=GFX1210 %s

ds_load_tr_b64 v[1:2], v0
// GFX1210: ds_load_tr8_b64 v[1:2], v0 ; encoding: [0x00,0x00,0xf4,0xdb,0x00,0x00,0x00,0x01]

ds_load_tr_b128 v[1:4], v0
// GFX1210: ds_load_tr16_b128 v[1:4], v0 ; encoding: [0x00,0x00,0xf0,0xdb,0x00,0x00,0x00,0x01]
