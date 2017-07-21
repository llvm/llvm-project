// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefixes=GCN,GFX9 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck -check-prefixes=GCN,VI %s

// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji %s 2>&1 | FileCheck -check-prefixes=ERR-SICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck -check-prefixes=ERR-SICIVI %s
// FIXME: pre-gfx9 errors should be more useful


// FIXME: These should parse to VOP2 encoding
v_add_u32 v1, v2, v3
// GFX9: v_add_u32_e64 v1, v2, v3        ; encoding: [0x01,0x00,0x34,0xd1,0x02,0x07,0x02,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_add_u32 v1, v2, s1
// GFX9: v_add_u32_e64 v1, v2, s1        ; encoding: [0x01,0x00,0x34,0xd1,0x02,0x03,0x00,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_add_u32 v1, s1, v2
// GFX9: v_add_u32_e64 v1, s1, v2        ; encoding: [0x01,0x00,0x34,0xd1,0x01,0x04,0x02,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_add_u32 v1, 4.0, v2
// GFX9: v_add_u32_e64 v1, 4.0, v2       ; encoding: [0x01,0x00,0x34,0xd1,0xf6,0x04,0x02,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_add_u32 v1, v2, 4.0
// GFX9: v_add_u32_e64 v1, v2, 4.0       ; encoding: [0x01,0x00,0x34,0xd1,0x02,0xed,0x01,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_add_u32_e32 v1, v2, v3
// GFX9: v_add_u32_e32 v1, v2, v3        ; encoding: [0x02,0x07,0x02,0x68]
// ERR-SICIVI: :19: error: invalid operand for instruction

v_add_u32_e32 v1, s1, v3
// GFX9: v_add_u32_e32 v1, s1, v3        ; encoding: [0x01,0x06,0x02,0x68]
// ERR-SICIVI: :19: error: invalid operand for instruction



v_sub_u32 v1, v2, v3
// GFX9: v_sub_u32_e64 v1, v2, v3        ; encoding: [0x01,0x00,0x35,0xd1,0x02,0x07,0x02,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_sub_u32 v1, v2, s1
// GFX9: v_sub_u32_e64 v1, v2, s1        ; encoding: [0x01,0x00,0x35,0xd1,0x02,0x03,0x00,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_sub_u32 v1, s1, v2
// GFX9: v_sub_u32_e64 v1, s1, v2        ; encoding: [0x01,0x00,0x35,0xd1,0x01,0x04,0x02,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_sub_u32 v1, 4.0, v2
// GFX9: v_sub_u32_e64 v1, 4.0, v2       ; encoding: [0x01,0x00,0x35,0xd1,0xf6,0x04,0x02,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_sub_u32 v1, v2, 4.0
// GFX9: v_sub_u32_e64 v1, v2, 4.0       ; encoding: [0x01,0x00,0x35,0xd1,0x02,0xed,0x01,0x00]
// ERR-SICIVI: :15: error: invalid operand for instruction

v_sub_u32_e32 v1, v2, v3
// GFX9: v_sub_u32_e32 v1, v2, v3        ; encoding: [0x02,0x07,0x02,0x6a]
// ERR-SICIVI: :19: error: invalid operand for instruction

v_sub_u32_e32 v1, s1, v3
// GFX9: v_sub_u32_e32 v1, s1, v3        ; encoding: [0x01,0x06,0x02,0x6a]
// ERR-SICIVI: :19: error: invalid operand for instruction



v_subrev_u32 v1, v2, v3
// GFX9: v_subrev_u32_e64 v1, v2, v3     ; encoding: [0x01,0x00,0x36,0xd1,0x02,0x07,0x02,0x00]
// ERR-SICIVI: :18: error: invalid operand for instruction

v_subrev_u32 v1, v2, s1
// GFX9: v_subrev_u32_e64 v1, v2, s1     ; encoding: [0x01,0x00,0x36,0xd1,0x02,0x03,0x00,0x00]
// ERR-SICIVI: :18: error: invalid operand for instruction

v_subrev_u32 v1, s1, v2
// GFX9: v_subrev_u32_e64 v1, s1, v2     ; encoding: [0x01,0x00,0x36,0xd1,0x01,0x04,0x02,0x00]
// ERR-SICIVI: :18: error: invalid operand for instruction

v_subrev_u32 v1, 4.0, v2
// GFX9: v_subrev_u32_e64 v1, 4.0, v2    ; encoding: [0x01,0x00,0x36,0xd1,0xf6,0x04,0x02,0x00]
// ERR-SICIVI: :18: error: invalid operand for instruction

v_subrev_u32 v1, v2, 4.0
// GFX9: v_subrev_u32_e64 v1, v2, 4.0    ; encoding: [0x01,0x00,0x36,0xd1,0x02,0xed,0x01,0x00]
// ERR-SICIVI: :18: error: invalid operand for instruction

v_subrev_u32_e32 v1, v2, v3
// GFX9: v_subrev_u32_e32 v1, v2, v3     ; encoding: [0x02,0x07,0x02,0x6c]
// ERR-SICIVI: :22: error: invalid operand for instruction

v_subrev_u32_e32 v1, s1, v3
// GFX9: v_subrev_u32_e32 v1, s1, v3     ; encoding: [0x01,0x06,0x02,0x6c]
// ERR-SICIVI: :22: error: invalid operand for instruction



v_add_u32 v1, vcc, v2, v3
// GCN: v_add_i32_e32 v1, vcc, v2, v3   ; encoding: [0x02,0x07,0x02,0x32]

v_add_u32 v1, s[0:1], v2, v3
// GCN: v_add_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x19,0xd1,0x02,0x07,0x02,0x00]
