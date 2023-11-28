// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding < %s | FileCheck --check-prefix=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

v_bitop3_b32_e64_dpp v5, v1, v2, v3 quad_perm:[3,2,1,0]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, v3 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x34,0xd6,0xfa,0x04,0x0e,0x04,0x01,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, v3 bitop3:161 quad_perm:[0,1,2,3]
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, v3 bitop3:0xa1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x04,0x34,0xd6,0xfa,0x04,0x0e,0x34,0x01,0xe4,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, v3 bitop3:0x27 row_mirror
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, v3 bitop3:0x27 row_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x04,0x34,0xd6,0xfa,0x04,0x0e,0xe4,0x01,0x40,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, v255 bitop3:100 row_half_mirror
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, v255 bitop3:0x64 row_half_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x04,0x34,0xd6,0xfa,0x04,0xfe,0x8f,0x01,0x41,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, s105 bitop3:0 row_shl:1
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, s105 row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x34,0xd6,0xfa,0x04,0xa6,0x01,0x01,0x01,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, vcc_hi bitop3:0x15 row_shl:15
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, vcc_hi bitop3:0x15 row_shl:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x02,0x34,0xd6,0xfa,0x04,0xae,0xa1,0x01,0x0f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, vcc_lo bitop3:63 row_shr:1
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, vcc_lo bitop3:0x3f row_shr:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x07,0x34,0xd6,0xfa,0x04,0xaa,0xe1,0x01,0x11,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, ttmp15 bitop3:0x24 row_shr:15
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, ttmp15 bitop3:0x24 row_shr:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x04,0x34,0xd6,0xfa,0x04,0xee,0x81,0x01,0x1f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, exec_hi bitop3:5 row_ror:1
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, exec_hi bitop3:5 row_ror:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x34,0xd6,0xfa,0x04,0xfe,0xa1,0x01,0x21,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, exec_lo bitop3:6 row_ror:15
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, exec_lo bitop3:6 row_ror:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x34,0xd6,0xfa,0x04,0xfa,0xc1,0x01,0x2f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, null bitop3:77 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, null bitop3:0x4d row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x01,0x34,0xd6,0xfa,0x04,0xf2,0xa9,0x01,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, -1 bitop3:88 row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, -1 bitop3:0x58 row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0x05,0x03,0x34,0xd6,0xfa,0x04,0x06,0x0b,0x01,0x5f,0x01,0x01]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v5, v1, v2, 0.5 bitop3:99 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 fi:0
// GFX1210: v_bitop3_b32_e64_dpp v5, v1, v2, 0.5 bitop3:0x63 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 ; encoding: [0x05,0x04,0x34,0xd6,0xfa,0x04,0xc2,0x6b,0x01,0x60,0x09,0x13]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b32_e64_dpp v255, v255, v255, src_scc bitop3:101 row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// GFX1210: v_bitop3_b32_e64_dpp v255, v255, v255, src_scc bitop3:0x65 row_xmask:15 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xff,0x04,0x34,0xd6,0xfa,0xfe,0xf7,0xab,0xff,0x6f,0x05,0x30]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, v3 quad_perm:[3,2,1,0]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, v3 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x33,0xd6,0xfa,0x04,0x0e,0x04,0x01,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, v3 bitop3:161 quad_perm:[0,1,2,3]
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, v3 bitop3:0xa1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x04,0x33,0xd6,0xfa,0x04,0x0e,0x34,0x01,0xe4,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, v3 bitop3:0x27 row_mirror
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, v3 bitop3:0x27 row_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x04,0x33,0xd6,0xfa,0x04,0x0e,0xe4,0x01,0x40,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, v3 bitop3:100 row_half_mirror
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, v3 bitop3:0x64 row_half_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x04,0x33,0xd6,0xfa,0x04,0x0e,0x8c,0x01,0x41,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, v255 bitop3:0 row_shl:1
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, v255 row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x33,0xd6,0xfa,0x04,0xfe,0x07,0x01,0x01,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, s105 bitop3:0x16 row_shl:15
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, s105 bitop3:0x16 row_shl:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x02,0x33,0xd6,0xfa,0x04,0xa6,0xc1,0x01,0x0f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, vcc_hi bitop3:63 row_shr:1
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, vcc_hi bitop3:0x3f row_shr:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x07,0x33,0xd6,0xfa,0x04,0xae,0xe1,0x01,0x11,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, vcc_lo bitop3:0x24 row_shr:15
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, vcc_lo bitop3:0x24 row_shr:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x04,0x33,0xd6,0xfa,0x04,0xaa,0x81,0x01,0x1f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, ttmp15 bitop3:5 row_ror:1
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, ttmp15 bitop3:5 row_ror:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x33,0xd6,0xfa,0x04,0xee,0xa1,0x01,0x21,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi bitop3:6 row_ror:15
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi bitop3:6 row_ror:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x33,0xd6,0xfa,0x04,0xfe,0xc1,0x01,0x2f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x33,0xd6,0xfa,0x04,0xfa,0x01,0x01,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo bitop3:77 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo bitop3:0x4d row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x01,0x33,0xd6,0xfa,0x04,0xfa,0xa9,0x01,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, null bitop3:88 row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, null bitop3:0x58 row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0x05,0x03,0x33,0xd6,0xfa,0x04,0xf2,0x09,0x01,0x5f,0x01,0x01]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, -1 bitop3:99 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 fi:0
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, -1 bitop3:0x63 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 ; encoding: [0x05,0x04,0x33,0xd6,0xfa,0x04,0x06,0x6b,0x01,0x60,0x09,0x13]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v255, v255, v255, src_scc bitop3:101 row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// GFX1210: v_bitop3_b16_e64_dpp v255, v255, v255, src_scc bitop3:0x65 row_xmask:15 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xff,0x04,0x33,0xd6,0xfa,0xfe,0xf7,0xab,0xff,0x6f,0x05,0x30]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi op_sel:[1,1,1,1] row_ror:15 row_mask:0xf bank_mask:0xf
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi op_sel:[1,1,1,1] row_ror:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x78,0x33,0xd6,0xfa,0x04,0xfe,0x01,0x01,0x2f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi bitop3:102 op_sel:[1,1,1,1] row_ror:15 row_mask:0xf bank_mask:0xf
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_hi bitop3:0x66 op_sel:[1,1,1,1] row_ror:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x7c,0x33,0xd6,0xfa,0x04,0xfe,0xc9,0x01,0x2f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo bitop3:103 op_sel:[1,0,0,0] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, exec_lo bitop3:0x67 op_sel:[1,0,0,0] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x0c,0x33,0xd6,0xfa,0x04,0xfa,0xe9,0x01,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, null bitop3:104 op_sel:[0,1,0,0] row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, null bitop3:0x68 op_sel:[0,1,0,0] row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0x05,0x15,0x33,0xd6,0xfa,0x04,0xf2,0x09,0x01,0x5f,0x01,0x01]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v5, v1, v2, -1 bitop3:104 op_sel:[0,0,1,0] row_xmask:0 row_mask:0x1 bank_mask:0x3
// GFX1210: v_bitop3_b16_e64_dpp v5, v1, v2, -1 bitop3:0x68 op_sel:[0,0,1,0] row_xmask:0 row_mask:0x1 bank_mask:0x3 ; encoding: [0x05,0x25,0x33,0xd6,0xfa,0x04,0x06,0x0b,0x01,0x60,0x01,0x13]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_bitop3_b16_e64_dpp v255, v255, v255, src_scc bitop3:104 op_sel:[0,0,0,1] row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:1 fi:1
// GFX1210: v_bitop3_b16_e64_dpp v255, v255, v255, src_scc bitop3:0x68 op_sel:[0,0,0,1] row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:1 fi:1 ; encoding: [0xff,0x45,0x33,0xd6,0xfa,0xfe,0xf7,0x0b,0xff,0x6f,0x0d,0x30]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] row_share:3
// GFX1210: v_lshl_add_u64_e64_dpp v[2:3], v[4:5], v7, v[8:9] row_share:3 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x52,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_lshl_add_u64 v[2:3], v[4:5], v4, v[2:3] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_lshl_add_u64_e64_dpp v[2:3], v[4:5], v4, v[2:3] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x52,0xd6,0xfa,0x08,0x0a,0x04,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX1210: v_fma_f64_e64_dpp v[4:5], v[2:3], v[6:7], v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x14,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_fma_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX12-ERR-NEXT:{{^}}                                         ^

v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX1210: v_div_fixup_f64_e64_dpp v[4:5], v[2:3], v[6:7], v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x28,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_div_fixup_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX12-ERR-NEXT:{{^}}                                               ^

v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX1210: v_div_fmas_f64_e64_dpp v[4:5], v[2:3], v[6:7], v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x38,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_div_fmas_f64 v[4:5], v[2:3], v[6:7], v[8:9] row_share:1
// GFX12-ERR-NEXT:{{^}}                                              ^

v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] row_share:1
// GFX1210: v_div_scale_f64_e64_dpp v[4:5], s2, v[2:3], v[6:7], v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x02,0xfd,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_div_scale_f64 v[4:5], s2, v[2:3], v[6:7], v[8:9] row_share:1
// GFX12-ERR-NEXT:{{^}}                                                   ^

v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] row_share:1
// GFX1210: v_mad_co_u64_u32_e64_dpp v[4:5], s2, v2, v6, v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x02,0xfe,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_mad_co_u64_u32 v[4:5], s2, v2, v6, v[8:9] row_share:1
// GFX12-ERR-NEXT:{{^}}                                            ^

v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] row_share:1
// GFX1210: v_mad_co_i64_i32_e64_dpp v[4:5], s2, v2, v6, v[8:9] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x02,0xff,0xd6,0xfa,0x0c,0x22,0x04,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_mad_co_i64_i32 v[4:5], s2, v2, v6, v[8:9] row_share:1
// GFX12-ERR-NEXT:{{^}}                                            ^

v_minimum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX1210: v_minimum_f64_e64_dpp v[4:5], v[2:3], v[6:7] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x41,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_minimum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX12-ERR-NEXT:{{^}}                                     ^

v_maximum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX1210: v_maximum_f64_e64_dpp v[4:5], v[2:3], v[6:7] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x42,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_maximum_f64 v[4:5], v[2:3], v[6:7] row_share:1
// GFX12-ERR-NEXT:{{^}}                                     ^

v_ldexp_f64 v[4:5], v[2:3], v6 row_share:1
// GFX1210: v_ldexp_f64_e64_dpp v[4:5], v[2:3], v6 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x2b,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_ldexp_f64 v[4:5], v[2:3], v6 row_share:1
// GFX12-ERR-NEXT:{{^}}                               ^

v_mul_lo_u32 v4, v2, v6 row_share:1
// GFX1210: v_mul_lo_u32_e64_dpp v4, v2, v6 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x2c,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_mul_lo_u32 v4, v2, v6 row_share:1
// GFX12-ERR-NEXT:{{^}}                        ^

v_mul_hi_u32 v4, v2, v6 row_share:1
// GFX1210: v_mul_hi_u32_e64_dpp v4, v2, v6 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x2d,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_mul_hi_u32 v4, v2, v6 row_share:1
// GFX12-ERR-NEXT:{{^}}                        ^

v_mul_hi_i32 v4, v2, v6 row_share:1
// GFX1210: v_mul_hi_i32_e64_dpp v4, v2, v6 row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x2e,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_mul_hi_i32 v4, v2, v6 row_share:1
// GFX12-ERR-NEXT:{{^}}                        ^

v_lshrrev_b64 v[4:5], v2, v[6:7] row_share:1
// GFX1210: v_lshrrev_b64_e64_dpp v[4:5], v2, v[6:7] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x3d,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_lshrrev_b64 v[4:5], v2, v[6:7] row_share:1
// GFX12-ERR-NEXT:{{^}}                                 ^

v_ashrrev_i64 v[4:5], v2, v[6:7] row_share:1
// GFX1210: v_ashrrev_i64_e64_dpp v[4:5], v2, v[6:7] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0x04,0x00,0x3e,0xd7,0xfa,0x0c,0x02,0x00,0x02,0x51,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX12-ERR-NEXT:{{^}}v_ashrrev_i64 v[4:5], v2, v[6:7] row_share:1
// GFX12-ERR-NEXT:{{^}}                                 ^

v_mad_u32 v2, v4, v7, v8 row_share:3 fi:1
// GFX1210: v_mad_u32_e64_dpp v2, v4, v7, v8 row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x35,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_u32 v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_mad_u32_e64_dpp v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x35,0xd6,0xfa,0x0e,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_max_i64 v[2:3], v[4:5], v[6:7] row_share:3 fi:1
// GFX1210: v_max_i64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x1b,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_max_i64 v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_max_i64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x1b,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_max_u64 v[2:3], v[4:5], v[6:7] row_share:3 fi:1
// GFX1210: v_max_u64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x19,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_max_u64 v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_max_u64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x19,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_min_i64 v[2:3], v[4:5], v[6:7] row_share:3 fi:1
// GFX1210: v_min_i64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x1a,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_min_i64 v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_min_i64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x1a,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_min_u64 v[2:3], v[4:5], v[6:7] row_share:3 fi:1
// GFX1210: v_min_u64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x18,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_min_u64 v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_min_u64_e64_dpp v[2:3], v[4:5], v[6:7] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x18,0xd7,0xfa,0x0c,0x02,0x00,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_nc_u64_u32 v[2:3], v4, v7, v[8:9] row_share:3 fi:1
// GFX1210: v_mad_nc_u64_u32_e64_dpp v[2:3], v4, v7, v[8:9] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0xfa,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_nc_u64_u32 v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_mad_nc_u64_u32_e64_dpp v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xfa,0xd6,0xfa,0x0a,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_nc_i64_i32 v[2:3], v4, v7, v[8:9] row_share:3 fi:1
// GFX1210: v_mad_nc_i64_i32_e64_dpp v[2:3], v4, v7, v[8:9] row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0xfb,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_nc_i64_i32 v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_mad_nc_i64_i32_e64_dpp v[2:3], v4, v5, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0xfb,0xd6,0xfa,0x0a,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_min_i32 v2, v4, v7, v8 quad_perm:[1,2,3,1]
// GFX1210: v_add_min_i32_e64_dpp v2, v4, v7, v8 quad_perm:[1,2,3,1] row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x60,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x79,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_min_i32 v2, v4, v7, v8 row_share:3 fi:1
// GFX1210: v_add_min_i32_e64_dpp v2, v4, v7, v8 row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x60,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_min_i32 v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_add_min_i32_e64_dpp v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x60,0xd6,0xfa,0x0e,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_max_i32 v2, v4, v7, v8 quad_perm:[3,2,1,0]
// GFX1210: v_add_max_i32_e64_dpp v2, v4, v7, v8 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x5e,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_max_i32 v2, v4, v7, v8 row_share:3 fi:1
// GFX1210: v_add_max_i32_e64_dpp v2, v4, v7, v8 row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x5e,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_max_i32 v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_add_max_i32_e64_dpp v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x5e,0xd6,0xfa,0x0e,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_min_u32 v2, v4, v7, v8 quad_perm:[3,2,1,0]
// GFX1210: v_add_min_u32_e64_dpp v2, v4, v7, v8 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x61,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_min_u32 v2, v4, v7, v8 row_share:3 fi:1
// GFX1210: v_add_min_u32_e64_dpp v2, v4, v7, v8 row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x61,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_min_u32 v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_add_min_u32_e64_dpp v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x61,0xd6,0xfa,0x0e,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_max_u32 v2, v4, v7, v8 quad_perm:[3,2,1,0]
// GFX1210: v_add_max_u32_e64_dpp v2, v4, v7, v8 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x5f,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_max_u32 v2, v4, v7, v8 row_share:3 fi:1
// GFX1210: v_add_max_u32_e64_dpp v2, v4, v7, v8 row_share:3 row_mask:0xf bank_mask:0xf fi:1 ; encoding: [0x02,0x00,0x5f,0xd6,0xfa,0x0e,0x22,0x04,0x04,0x53,0x05,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_max_u32 v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_add_max_u32_e64_dpp v2, v4, v7, 1 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x02,0x00,0x5f,0xd6,0xfa,0x0e,0x06,0x02,0x04,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 quad_perm:[3,2,1,0]
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 quad_perm:[0,1,2,3]
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0xe4,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_mirror
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x40,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_half_mirror
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_half_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x41,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_shl:1
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x01,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_shl:15
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_shl:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x0f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_shr:1
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_shr:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x11,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_shr:15
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_shr:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x1f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_ror:1
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_ror:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x21,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_ror:15
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_ror:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x2f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x00,0x01,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x08,0x01,0x5f,0x01,0x01]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 fi:0
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v5, v1, v2 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 ; encoding: [0x05,0x00,0x6d,0xd7,0xfa,0x04,0x02,0x10,0x01,0x60,0x09,0x13]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32_e64_dpp v255, -|v255|, v255 clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// GFX1210: v_cvt_pk_bf16_f32_e64_dpp v255, -|v255|, v255 clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xff,0x81,0x6d,0xd7,0xfa,0xfe,0x03,0x38,0xff,0x6f,0x05,0x30]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v3 quad_perm:[3,2,1,0]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v3 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6e,0xd7,0xfa,0x04,0x0e,0x04,0x01,0x1b,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v3 quad_perm:[0,1,2,3]
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6e,0xd7,0xfa,0x04,0x0e,0x04,0x01,0xe4,0x00,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v3 row_mirror
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v3 row_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6e,0xd7,0xfa,0x04,0x0e,0x04,0x01,0x40,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v255 row_half_mirror
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, v255 row_half_mirror row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6e,0xd7,0xfa,0x04,0xfe,0x07,0x01,0x41,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, s105 row_shl:1
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, s105 row_shl:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6e,0xd7,0xfa,0x04,0xa6,0x01,0x01,0x01,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, vcc_hi row_shl:15
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, vcc_hi row_shl:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6e,0xd7,0xfa,0x04,0xae,0x01,0x01,0x0f,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, vcc_lo row_shr:1
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, v2, vcc_lo row_shr:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x6e,0xd7,0xfa,0x04,0xaa,0x01,0x01,0x11,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, -|v2|, exec_hi row_ror:1
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, -|v2|, exec_hi row_ror:1 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x02,0x6e,0xd7,0xfa,0x04,0xfe,0x41,0x01,0x21,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, -|v1|, -|v2|, null row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, -|v1|, -|v2|, null row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x03,0x6e,0xd7,0xfa,0x04,0xf2,0x61,0x01,0x50,0x01,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, -|v1|, v2, -1 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, -|v1|, v2, -1 mul:2 row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0x05,0x01,0x6e,0xd7,0xfa,0x04,0x06,0x2b,0x01,0x5f,0x01,0x01]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, -|v2|, 5 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 fi:0
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v5, v1, -|v2|, 5 mul:4 row_xmask:0 row_mask:0x1 bank_mask:0x3 bound_ctrl:1 ; encoding: [0x05,0x02,0x6e,0xd7,0xfa,0x04,0x16,0x52,0x01,0x60,0x09,0x13]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_sr_pk_bf16_f32_e64_dpp v255, -|v255|, -|v255|, src_scc clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 bound_ctrl:0 fi:1
// GFX1210: v_cvt_sr_pk_bf16_f32_e64_dpp v255, -|v255|, -|v255|, src_scc clamp div:2 row_xmask:15 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xff,0x83,0x6e,0xd7,0xfa,0xfe,0xf7,0x7b,0xff,0x6f,0x05,0x30]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
