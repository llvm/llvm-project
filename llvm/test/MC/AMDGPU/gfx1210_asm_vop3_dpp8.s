// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding < %s | FileCheck --check-prefix=GFX1210 %s

v_bitop3_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x34,0xd6,0xe9,0x04,0x0e,0x04,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, v255 bitop3:161 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, v255 bitop3:0xa1 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x04,0x34,0xd6,0xe9,0x04,0xfe,0x37,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, s105 bitop3:0x27 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, s105 bitop3:0x27 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x04,0x34,0xd6,0xe9,0x04,0xa6,0xe1,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, vcc_hi bitop3:100 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, vcc_hi bitop3:0x64 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x04,0x34,0xd6,0xe9,0x04,0xae,0x89,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, vcc_lo bitop3:0 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, vcc_lo dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x34,0xd6,0xe9,0x04,0xaa,0x01,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, ttmp15 bitop3:0x15 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, ttmp15 bitop3:0x15 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x02,0x34,0xd6,0xe9,0x04,0xee,0xa1,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, exec_hi bitop3:63 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, exec_hi bitop3:0x3f dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x07,0x34,0xd6,0xe9,0x04,0xfe,0xe1,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, exec_lo bitop3:0x24 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, exec_lo bitop3:0x24 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x04,0x34,0xd6,0xe9,0x04,0xfa,0x81,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, null bitop3:5 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, null bitop3:5 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x34,0xd6,0xe9,0x04,0xf2,0xa1,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, -1 bitop3:6 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, -1 bitop3:6 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x34,0xd6,0xe9,0x04,0x06,0xc3,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v5, v1, v2, 0.5 bitop3:77 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, 0.5 bitop3:0x4d dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0x05,0x01,0x34,0xd6,0xea,0x04,0xc2,0xab,0x01,0x77,0x39,0x05]

v_bitop3_b32_e64_dpp v255, v255, v255, src_scc bitop3:88 dpp8:[0,0,0,0,0,0,0,0] fi:0
// GFX1210: v_bitop3_b32_e64_dpp v255, v255, v255, src_scc bitop3:0x58 dpp8:[0,0,0,0,0,0,0,0] ; encoding: [0xff,0x03,0x34,0xd6,0xe9,0xfe,0xf7,0x0b,0xff,0x00,0x00,0x00]

v_bitop3_b16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x33,0xd6,0xe9,0x04,0x0e,0x04,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, v255 bitop3:161 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, v255 bitop3:0xa1 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x04,0x33,0xd6,0xe9,0x04,0xfe,0x37,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, s105 bitop3:0x27 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, s105 bitop3:0x27 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x04,0x33,0xd6,0xe9,0x04,0xa6,0xe1,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, vcc_hi bitop3:100 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, vcc_hi bitop3:0x64 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x04,0x33,0xd6,0xe9,0x04,0xae,0x89,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, vcc_lo bitop3:0 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, vcc_lo dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x33,0xd6,0xe9,0x04,0xaa,0x01,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, ttmp15 bitop3:15 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, ttmp15 bitop3:0xf dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x01,0x33,0xd6,0xe9,0x04,0xee,0xe1,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi bitop3:63 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi bitop3:0x3f dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x07,0x33,0xd6,0xe9,0x04,0xfe,0xe1,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo bitop3:0x24 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo bitop3:0x24 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x04,0x33,0xd6,0xe9,0x04,0xfa,0x81,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, null bitop3:5 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, null bitop3:5 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x33,0xd6,0xe9,0x04,0xf2,0xa1,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, -1 bitop3:6 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, -1 bitop3:6 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0x05,0x00,0x33,0xd6,0xea,0x04,0x06,0xc3,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v255, v255, v255, src_scc bitop3:77 dpp8:[0,0,0,0,0,0,0,0] fi:0
// GFX1210: v_bitop3_b16_e64_dpp v255, v255, v255, src_scc bitop3:0x4d dpp8:[0,0,0,0,0,0,0,0] ; encoding: [0xff,0x01,0x33,0xd6,0xe9,0xfe,0xf7,0xab,0xff,0x00,0x00,0x00]

v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi op_sel:[1,1,1,1] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi op_sel:[1,1,1,1] dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x78,0x33,0xd6,0xe9,0x04,0xfe,0x01,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi bitop3:88 op_sel:[1,1,1,1] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi bitop3:0x58 op_sel:[1,1,1,1] dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x7b,0x33,0xd6,0xe9,0x04,0xfe,0x09,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo bitop3:99 op_sel:[1,0,0,0] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo bitop3:0x63 op_sel:[1,0,0,0] dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x0c,0x33,0xd6,0xe9,0x04,0xfa,0x69,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, null op_sel:[0,1,0,0] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, null op_sel:[0,1,0,0] dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x10,0x33,0xd6,0xe9,0x04,0xf2,0x01,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v5, v1, v2, -1 bitop3:102 op_sel:[0,0,1,0] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, -1 bitop3:0x66 op_sel:[0,0,1,0] dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x24,0x33,0xd6,0xe9,0x04,0x06,0xcb,0x01,0x77,0x39,0x05]

v_bitop3_b16_e64_dpp v255, v255, v255, src_scc bitop3:103 op_sel:[0,0,0,1] dpp8:[0,0,0,0,0,0,0,0] fi:1
// GFX1210: v_bitop3_b16_e64_dpp v255, v255, v255, src_scc bitop3:0x67 op_sel:[0,0,0,1] dpp8:[0,0,0,0,0,0,0,0] fi:1 ; encoding: [0xff,0x44,0x33,0xd6,0xea,0xfe,0xf7,0xeb,0xff,0x00,0x00,0x00]

v_add_min_i32 v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_add_min_i32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x60,0xd6,0xe9,0x04,0x0e,0x04,0x01,0x77,0x39,0x05]

v_add_min_i32 v5, v1, 42, s3 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_add_min_i32_e64_dpp v5, v1, 42, s3 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0x05,0x00,0x60,0xd6,0xea,0x54,0x0d,0x00,0x01,0x77,0x39,0x05]

v_add_max_i32 v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_add_max_i32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x5e,0xd6,0xe9,0x04,0x0e,0x04,0x01,0x77,0x39,0x05]

v_add_max_i32 v5, v1, 42, s3 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_add_max_i32_e64_dpp v5, v1, 42, s3 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0x05,0x00,0x5e,0xd6,0xea,0x54,0x0d,0x00,0x01,0x77,0x39,0x05]

v_add_min_u32 v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_add_min_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x61,0xd6,0xe9,0x04,0x0e,0x04,0x01,0x77,0x39,0x05]

v_add_min_u32 v5, v1, 42, s3 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_add_min_u32_e64_dpp v5, v1, 42, s3 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0x05,0x00,0x61,0xd6,0xea,0x54,0x0d,0x00,0x01,0x77,0x39,0x05]

v_add_max_u32 v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_add_max_u32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x5f,0xd6,0xe9,0x04,0x0e,0x04,0x01,0x77,0x39,0x05]

v_add_max_u32 v5, v1, 42, s3 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_add_max_u32_e64_dpp v5, v1, 42, s3 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0x05,0x00,0x5f,0xd6,0xea,0x54,0x0d,0x00,0x01,0x77,0x39,0x05]

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x6d,0xd7,0xe9,0x04,0x02,0x00,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 mul:2 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 mul:2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x6d,0xd7,0xe9,0x04,0x02,0x08,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 mul:4 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 mul:4 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0x05,0x00,0x6d,0xd7,0xea,0x04,0x02,0x10,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v255, -|v255|, v255 clamp div:2 dpp8:[0,0,0,0,0,0,0,0] fi:0
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v255, -|v255|, v255 clamp div:2 dpp8:[0,0,0,0,0,0,0,0] ; encoding: [0xff,0x81,0x6d,0xd7,0xe9,0xfe,0x03,0x38,0xff,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x6e,0xd7,0xe9,0x04,0x0e,0x04,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v255 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x6e,0xd7,0xe9,0x04,0xfe,0x07,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, s105 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, s105 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x6e,0xd7,0xe9,0x04,0xa6,0x01,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, vcc_hi dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, vcc_hi dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x6e,0xd7,0xe9,0x04,0xae,0x01,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, vcc_lo dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, vcc_lo dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x6e,0xd7,0xe9,0x04,0xaa,0x01,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, -|v2|, exec_hi dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, -|v2|, exec_hi dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x02,0x6e,0xd7,0xe9,0x04,0xfe,0x41,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, -|v1|, -|v2|, null dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, -|v1|, -|v2|, null dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x03,0x6e,0xd7,0xe9,0x04,0xf2,0x61,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, -|v1|, v2, -1 mul:2 dpp8:[7,6,5,4,3,2,1,0]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, -|v1|, v2, -1 mul:2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x01,0x6e,0xd7,0xe9,0x04,0x06,0x2b,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, -|v2|, 5 mul:4 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, -|v2|, 5 mul:4 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0x05,0x02,0x6e,0xd7,0xea,0x04,0x16,0x52,0x01,0x77,0x39,0x05]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v255, -|v255|, -|v255|, src_scc clamp div:2 dpp8:[0,0,0,0,0,0,0,0] fi:0
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v255, -|v255|, -|v255|, src_scc clamp div:2 dpp8:[0,0,0,0,0,0,0,0] ; encoding: [0xff,0x83,0x6e,0xd7,0xe9,0xfe,0xf7,0x7b,0xff,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
