// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1200 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX12 %s

v_pk_min_f16 v0, v1, v2
// GFX12: v_pk_min_num_f16 v0, v1, v2             ; encoding: [0x00,0x40,0x1b,0xcc,0x01,0x05,0x02,0x18]

v_pk_max_f16 v0, v1, v2
// GFX12: v_pk_max_num_f16 v0, v1, v2             ; encoding: [0x00,0x40,0x1c,0xcc,0x01,0x05,0x02,0x18]
