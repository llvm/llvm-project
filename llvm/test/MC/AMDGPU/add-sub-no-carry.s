// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefix=GFX9 %s

// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji %s 2>&1 | FileCheck --check-prefix=ERR-VI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck --check-prefix=ERR-SICI --implicit-check-not=error: %s
// FIXME: pre-gfx9 errors should be more useful


v_add_u32 v1, v2, v3
// GFX9: v_add_u32_e32 v1, v2, v3       ; encoding: [0x02,0x07,0x02,0x68]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_u32 v1, v2, s1
// GFX9: v_add_u32_e64 v1, v2, s1        ; encoding: [0x01,0x00,0x34,0xd1,0x02,0x03,0x00,0x00]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_u32 v1, s1, v2
// GFX9: v_add_u32_e32 v1, s1, v2        ; encoding: [0x01,0x04,0x02,0x68]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_u32 v1, 4.0, v2
// GFX9: v_add_u32_e32 v1, 4.0, v2       ; encoding: [0xf6,0x04,0x02,0x68]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_u32 v1, v2, 4.0
// GFX9: v_add_u32_e64 v1, v2, 4.0       ; encoding: [0x01,0x00,0x34,0xd1,0x02,0xed,0x01,0x00]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_u32_e32 v1, v2, v3
// GFX9: v_add_u32_e32 v1, v2, v3        ; encoding: [0x02,0x07,0x02,0x68]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_u32_e32 v1, s1, v3
// GFX9: v_add_u32_e32 v1, s1, v3        ; encoding: [0x01,0x06,0x02,0x68]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode



v_sub_u32 v1, v2, v3
// GFX9: v_sub_u32_e32 v1, v2, v3        ; encoding: [0x02,0x07,0x02,0x6a]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_u32 v1, v2, s1
// GFX9: v_sub_u32_e64 v1, v2, s1        ; encoding: [0x01,0x00,0x35,0xd1,0x02,0x03,0x00,0x00]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_u32 v1, s1, v2
// GFX9: v_sub_u32_e32 v1, s1, v2        ; encoding: [0x01,0x04,0x02,0x6a]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_u32 v1, 4.0, v2
// GFX9: v_sub_u32_e32 v1, 4.0, v2       ; encoding: [0xf6,0x04,0x02,0x6a]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_u32 v1, v2, 4.0
// GFX9: v_sub_u32_e64 v1, v2, 4.0       ; encoding: [0x01,0x00,0x35,0xd1,0x02,0xed,0x01,0x00]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_u32_e32 v1, v2, v3
// GFX9: v_sub_u32_e32 v1, v2, v3        ; encoding: [0x02,0x07,0x02,0x6a]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_u32_e32 v1, s1, v3
// GFX9: v_sub_u32_e32 v1, s1, v3        ; encoding: [0x01,0x06,0x02,0x6a]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode



v_subrev_u32 v1, v2, v3
// GFX9: v_subrev_u32_e32 v1, v2, v3     ; encoding: [0x02,0x07,0x02,0x6c]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_u32 v1, v2, s1
// GFX9: v_subrev_u32_e64 v1, v2, s1     ; encoding: [0x01,0x00,0x36,0xd1,0x02,0x03,0x00,0x00]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_u32 v1, s1, v2
// GFX9: v_subrev_u32_e32 v1, s1, v2     ; encoding: [0x01,0x04,0x02,0x6c]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_u32 v1, 4.0, v2
// GFX9: v_subrev_u32_e32 v1, 4.0, v2    ; encoding: [0xf6,0x04,0x02,0x6c]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_u32 v1, v2, 4.0
// GFX9: v_subrev_u32_e64 v1, v2, 4.0    ; encoding: [0x01,0x00,0x36,0xd1,0x02,0xed,0x01,0x00]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_u32_e32 v1, v2, v3
// GFX9: v_subrev_u32_e32 v1, v2, v3     ; encoding: [0x02,0x07,0x02,0x6c]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_u32_e32 v1, s1, v3
// GFX9: v_subrev_u32_e32 v1, s1, v3     ; encoding: [0x01,0x06,0x02,0x6c]
// ERR-SICI: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// ERR-VI: :[[@LINE-3]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
