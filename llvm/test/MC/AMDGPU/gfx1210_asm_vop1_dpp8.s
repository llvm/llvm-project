// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefixes=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

v_tanh_f32 v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_tanh_f32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x3c,0x0a,0x7e,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32 v5, v1 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_tanh_f32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0xea,0x3c,0x0a,0x7e,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f32 v255, v255 dpp8:[0,0,0,0,0,0,0,0] fi:0
// GFX1210: v_tanh_f32_dpp v255, v255 dpp8:[0,0,0,0,0,0,0,0] ; encoding: [0xe9,0x3c,0xfe,0x7f,0xff,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16 v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_tanh_f16_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x3e,0x0a,0x7e,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16 v5, v1 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_tanh_f16_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0xea,0x3e,0x0a,0x7e,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_f16 v127, v127 dpp8:[0,0,0,0,0,0,0,0] fi:0
// GFX1210: v_tanh_f16_dpp v127, v127 dpp8:[0,0,0,0,0,0,0,0] ; encoding: [0xe9,0x3e,0xfe,0x7e,0x7f,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16 v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_tanh_bf16_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x94,0x0a,0x7e,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16 v5, v1 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_tanh_bf16_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0xea,0x94,0x0a,0x7e,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_tanh_bf16 v127, v127 dpp8:[0,0,0,0,0,0,0,0] fi:0
// GFX1210: v_tanh_bf16_dpp v127, v127 dpp8:[0,0,0,0,0,0,0,0] ; encoding: [0xe9,0x94,0xfe,0x7e,0x7f,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
