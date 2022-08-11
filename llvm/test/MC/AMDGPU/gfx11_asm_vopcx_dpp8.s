// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s

v_cmpx_f_i32_dpp v0, v2 dpp8:[7,6,5,3,4,2,1,0] fi:1
// GFX11: encoding: [0xea,0x04,0x80,0x7d,0x00,0x77,0x47,0x05]

v_cmpx_t_f32_dpp v255, v2 dpp8:[7,6,5,3,4,2,1,0]
// GFX11: encoding: [0xe9,0x04,0x3e,0x7d,0xff,0x77,0x47,0x05]

v_cmpx_class_f16_dpp v12, v101 dpp8:[7,6,5,3,4,2,1,0]
// GFX11: encoding: [0xe9,0xca,0xfa,0x7d,0x0c,0x77,0x47,0x05]
