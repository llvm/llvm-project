// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=-real-true16,+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=-real-true16,-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s

v_floor_f16 v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: encoding: [0xe9,0xb6,0x0a,0x7e,0x01,0x77,0x39,0x05]

v_floor_f16 v5, v1 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX11: encoding: [0xea,0xb6,0x0a,0x7e,0x01,0x77,0x39,0x05]

v_floor_f16 v127, v127 dpp8:[0,0,0,0,0,0,0,0] fi:0
// GFX11: encoding: [0xe9,0xb6,0xfe,0x7e,0x7f,0x00,0x00,0x00]

v_ceil_f16 v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: encoding: [0xe9,0xb8,0x0a,0x7e,0x01,0x77,0x39,0x05]

v_ceil_f16 v5, v1 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX11: encoding: [0xea,0xb8,0x0a,0x7e,0x01,0x77,0x39,0x05]

v_ceil_f16 v127, v127 dpp8:[0,0,0,0,0,0,0,0] fi:0
// GFX11: encoding: [0xe9,0xb8,0xfe,0x7e,0x7f,0x00,0x00,0x00]
