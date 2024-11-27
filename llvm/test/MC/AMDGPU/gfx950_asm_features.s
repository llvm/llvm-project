// RUN: llvm-mc -triple=amdgcn -mcpu=gfx950 -show-encoding %s | FileCheck --check-prefix=GFX950 --strict-whitespace %s
// xUN: not llvm-mc -triple=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck --check-prefixes=NOT-GFX950,GFX940 --implicit-check-not=error: %s
// xUN: not llvm-mc -triple=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefixes=NOT-GFX950,GFX90A --implicit-check-not=error: %s
// xUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=NOT-GFX950,GFX10 --implicit-check-not=error: %s

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: global_load_lds_dwordx3 v[2:3], off     ; encoding: [0x00,0x80,0xf8,0xdd,0x02,0x00,0x7f,0x00]

global_load_lds_dwordx3 v[2:3], off

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: global_load_lds_dwordx3 v[2:3], off sc0 nt sc1 ; encoding: [0x00,0x80,0xfb,0xdf,0x02,0x00,0x7f,0x00]
global_load_lds_dwordx3 v[2:3], off sc0 nt sc1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: global_load_lds_dwordx3 v[2:3], off offset:4 ; encoding: [0x04,0x80,0xf8,0xdd,0x02,0x00,0x7f,0x00]
global_load_lds_dwordx3 v[2:3], off offset:4

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: global_load_lds_dwordx3 v2, s[4:5] offset:4 ; encoding: [0x04,0x80,0xf8,0xdd,0x02,0x00,0x04,0x00]
global_load_lds_dwordx3 v2, s[4:5] offset:4

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: global_load_lds_dwordx4 v[2:3], off     ; encoding: [0x00,0x80,0xf4,0xdd,0x02,0x00,0x7f,0x00]
global_load_lds_dwordx4 v[2:3], off

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: global_load_lds_dwordx4 v[2:3], off sc0 nt sc1 ; encoding: [0x00,0x80,0xf7,0xdf,0x02,0x00,0x7f,0x00]
global_load_lds_dwordx4 v[2:3], off sc0 nt sc1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: global_load_lds_dwordx4 v[2:3], off offset:4 ; encoding: [0x04,0x80,0xf4,0xdd,0x02,0x00,0x7f,0x00]
global_load_lds_dwordx4 v[2:3], off offset:4

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: global_load_lds_dwordx4 v2, s[4:5] offset:4 ; encoding: [0x04,0x80,0xf4,0xdd,0x02,0x00,0x04,0x00]
global_load_lds_dwordx4 v2, s[4:5] offset:4


// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e32 v1, v2        ; encoding: [0x02,0xb3,0x02,0x7e]
v_permlane16_swap_b32 v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e32 v1, v2        ; encoding: [0x02,0xb3,0x02,0x7e]
v_permlane16_swap_b32_e32 v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e64 v1, v2        ; encoding: [0x01,0x00,0x99,0xd1,0x02,0x01,0x00,0x00]
v_permlane16_swap_b32_e64 v1, v2

// FIXME: Parsed as bound_ctrl:1?
// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e64 v1, v2 bound_ctrl:1 ; encoding: [0x01,0x10,0x99,0xd1,0x02,0x01,0x00,0x00]
v_permlane16_swap_b32 v1, v2 bound_ctrl:0

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e64 v1, v2        ; encoding: [0x01,0x00,0x99,0xd1,0x02,0x01,0x00,0x00]
v_permlane16_swap_b32 v1, v2 fi:0

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e64 v1, v2 bound_ctrl:1 ; encoding: [0x01,0x10,0x99,0xd1,0x02,0x01,0x00,0x00]
v_permlane16_swap_b32 v1, v2 bound_ctrl:1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e64 v1, v2 fi:1   ; encoding: [0x01,0x08,0x99,0xd1,0x02,0x01,0x00,0x00]
v_permlane16_swap_b32 v1, v2 fi:1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e64 v1, v2 bound_ctrl:1 fi:1 ; encoding: [0x01,0x18,0x99,0xd1,0x02,0x01,0x00,0x00]
v_permlane16_swap_b32 v1, v2 bound_ctrl:1 fi:1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e64 v1, v2 bound_ctrl:1 fi:1 ; encoding: [0x01,0x18,0x99,0xd1,0x02,0x01,0x00,0x00]
v_permlane16_swap_b32_e64 v1, v2 bound_ctrl:1 fi:1

// FIXME: Swapped order not accepted
// v_permlane16_swap_b32 v1, v2 fi:1 bound_ctrl:1


// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e32 v1, v2        ; encoding: [0x02,0xb5,0x02,0x7e]
v_permlane32_swap_b32 v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e32 v1, v2        ; encoding: [0x02,0xb5,0x02,0x7e]
v_permlane32_swap_b32_e32 v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e64 v1, v2        ; encoding: [0x01,0x00,0x9a,0xd1,0x02,0x01,0x00,0x00]
v_permlane32_swap_b32_e64 v1, v2

// FIXME: Parsed as bound_ctrl:1?
// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e64 v1, v2 bound_ctrl:1 ; encoding: [0x01,0x10,0x9a,0xd1,0x02,0x01,0x00,0x00]
v_permlane32_swap_b32 v1, v2 bound_ctrl:0

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e64 v1, v2        ; encoding: [0x01,0x00,0x9a,0xd1,0x02,0x01,0x00,0x00]
v_permlane32_swap_b32 v1, v2 fi:0

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e64 v1, v2 bound_ctrl:1 ; encoding: [0x01,0x10,0x9a,0xd1,0x02,0x01,0x00,0x00]
v_permlane32_swap_b32 v1, v2 bound_ctrl:1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e64 v1, v2 fi:1   ; encoding: [0x01,0x08,0x9a,0xd1,0x02,0x01,0x00,0x00]
v_permlane32_swap_b32 v1, v2 fi:1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e64 v1, v2 bound_ctrl:1 fi:1 ; encoding: [0x01,0x18,0x9a,0xd1,0x02,0x01,0x00,0x00]
v_permlane32_swap_b32 v1, v2 bound_ctrl:1 fi:1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e64 v1, v2 bound_ctrl:1 fi:1 ; encoding: [0x01,0x18,0x9a,0xd1,0x02,0x01,0x00,0x00]
v_permlane32_swap_b32_e64 v1, v2 bound_ctrl:1 fi:1

// FIXME: Swapped order not accepted
// v_permlane32_swap_b32 v1, v2 fi:1 bound_ctrl:1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, v2, v3       ; encoding: [0x01,0x00,0x4a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x4a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x4a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x4a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x4a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x4a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x4a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x4a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, v2, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, s1, v3       ; encoding: [0x01,0x00,0x4a,0xd2,0x01,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, s1, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, s2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x4a,0xd2,0x02,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, s2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, s3, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x4a,0xd2,0x03,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, s3, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, s4, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x4a,0xd2,0x04,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, s4, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, s1, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x4a,0xd2,0x01,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, s1, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, s2, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x4a,0xd2,0x02,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, s2, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, s3, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x4a,0xd2,0x03,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, s3, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, s4, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x4a,0xd2,0x04,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, s4, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, 11, v3       ; encoding: [0x01,0x00,0x4a,0xd2,0x8b,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, 11, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, 22, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x4a,0xd2,0x96,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, 22, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, 33, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x4a,0xd2,0xa1,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, 33, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, 44, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x4a,0xd2,0xac,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, 44, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, 11, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x4a,0xd2,0x8b,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, 11, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, 22, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x4a,0xd2,0x96,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, 22, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, 33, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x4a,0xd2,0xa1,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, 33, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_fp8 v1, 44, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x4a,0xd2,0xac,0x06,0x02,0x00]
v_cvt_scalef32_f16_fp8 v1, 44, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, v2, v3       ; encoding: [0x01,0x00,0x3b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x3b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x3b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x3b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x3b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x3b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x3b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x3b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, v2, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, s1, v3       ; encoding: [0x01,0x00,0x3b,0xd2,0x01,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, s1, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, s2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x3b,0xd2,0x02,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, s2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, s3, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x3b,0xd2,0x03,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, s3, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, s4, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x3b,0xd2,0x04,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, s4, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, s1, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x3b,0xd2,0x01,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, s1, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, s2, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x3b,0xd2,0x02,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, s2, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, s3, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x3b,0xd2,0x03,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, s3, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, s4, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x3b,0xd2,0x04,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, s4, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, 11, v3       ; encoding: [0x01,0x00,0x3b,0xd2,0x8b,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, 11, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, 22, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x3b,0xd2,0x96,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, 22, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, 33, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x3b,0xd2,0xa1,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, 33, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, 44, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x3b,0xd2,0xac,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, 44, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, 11, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x3b,0xd2,0x8b,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, 11, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, 22, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x3b,0xd2,0x96,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, 22, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, 33, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x3b,0xd2,0xa1,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, 33, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_fp8 v1, 44, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x3b,0xd2,0xac,0x06,0x02,0x00]
v_cvt_scalef32_f32_fp8 v1, 44, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, v2, v3       ; encoding: [0x01,0x00,0x4b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x4b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x4b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x4b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x4b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x4b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x4b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x4b,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, v2, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, s1, v3       ; encoding: [0x01,0x00,0x4b,0xd2,0x01,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, s1, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, s2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x4b,0xd2,0x02,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, s2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, s3, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x4b,0xd2,0x03,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, s3, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, s4, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x4b,0xd2,0x04,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, s4, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, s1, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x4b,0xd2,0x01,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, s1, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, s2, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x4b,0xd2,0x02,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, s2, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, s3, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x4b,0xd2,0x03,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, s3, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, s4, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x4b,0xd2,0x04,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, s4, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, 11, v3       ; encoding: [0x01,0x00,0x4b,0xd2,0x8b,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, 11, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, 22, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x4b,0xd2,0x96,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, 22, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, 33, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x4b,0xd2,0xa1,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, 33, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, 44, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x4b,0xd2,0xac,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, 44, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, 11, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x4b,0xd2,0x8b,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, 11, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, 22, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x4b,0xd2,0x96,0x06,0x02,0x00]
 v_cvt_scalef32_f16_bf8 v1, 22, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, 33, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x4b,0xd2,0xa1,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, 33, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f16_bf8 v1, 44, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x4b,0xd2,0xac,0x06,0x02,0x00]
v_cvt_scalef32_f16_bf8 v1, 44, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, v2, v3       ; encoding: [0x01,0x00,0x3c,0xd2,0x02,0x07,0x02,0x00]
 v_cvt_scalef32_f32_bf8 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x3c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x3c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x3c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x3c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x3c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x3c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x3c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, v2, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, s1, v3       ; encoding: [0x01,0x00,0x3c,0xd2,0x01,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, s1, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, s2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x3c,0xd2,0x02,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, s2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, s3, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x3c,0xd2,0x03,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, s3, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, s4, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x3c,0xd2,0x04,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, s4, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, s1, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x3c,0xd2,0x01,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, s1, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, s2, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x3c,0xd2,0x02,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, s2, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, s3, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x3c,0xd2,0x03,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, s3, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, s4, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x3c,0xd2,0x04,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, s4, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, 11, v3       ; encoding: [0x01,0x00,0x3c,0xd2,0x8b,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, 11, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, 22, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x3c,0xd2,0x96,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, 22, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, 33, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x3c,0xd2,0xa1,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, 33, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, 44, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x3c,0xd2,0xac,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, 44, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, 11, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x3c,0xd2,0x8b,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, 11, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, 22, v3 op_sel:[1,0,1] ; encoding: [0x01,0x48,0x3c,0xd2,0x96,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, 22, v3 op_sel:[1,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, 33, v3 op_sel:[0,1,1] ; encoding: [0x01,0x50,0x3c,0xd2,0xa1,0x06,0x02,0x00]
 v_cvt_scalef32_f32_bf8 v1, 33, v3 op_sel:[0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_f32_bf8 v1, 44, v3 op_sel:[1,1,1] ; encoding: [0x01,0x58,0x3c,0xd2,0xac,0x06,0x02,0x00]
v_cvt_scalef32_f32_bf8 v1, 44, v3 op_sel:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f32 v1, v1, v2, v3 ; encoding: [0x01,0x00,0x35,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_pk_fp8_f32 v1, v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f32 v1, v1, -v2, |v3| ; encoding: [0x01,0x04,0x35,0xd2,0x01,0x05,0x0e,0x44]
v_cvt_scalef32_pk_fp8_f32 v1, v1, -v2, |v3|

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f32 v1, v1, s2, 3 ; encoding: [0x01,0x00,0x35,0xd2,0x01,0x05,0x0c,0x02]
v_cvt_scalef32_pk_fp8_f32 v1, v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f32 v1, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x01,0x40,0x35,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_pk_fp8_f32 v1, v1, v2, v3 op_sel:[0,0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f32 v1, v1, -v2, |v3| op_sel:[0,0,0,1] ; encoding: [0x01,0x44,0x35,0xd2,0x01,0x05,0x0e,0x44]
v_cvt_scalef32_pk_fp8_f32 v1, v1, -v2, |v3| op_sel:[0,0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f32 v1, v1, s2, 3 op_sel:[0,0,0,1] ; encoding: [0x01,0x40,0x35,0xd2,0x01,0x05,0x0c,0x02]
v_cvt_scalef32_pk_fp8_f32 v1, v1, s2, 3 op_sel:[0,0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f32 v1, v1, v2, v3 ; encoding: [0x01,0x00,0x36,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_pk_bf8_f32 v1, v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f32 v1, v1, -v2, |v3| ; encoding: [0x01,0x04,0x36,0xd2,0x01,0x05,0x0e,0x44]
v_cvt_scalef32_pk_bf8_f32 v1, v1, -v2, |v3|

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f32 v1, v1, s2, 3 ; encoding: [0x01,0x00,0x36,0xd2,0x01,0x05,0x0c,0x02]
v_cvt_scalef32_pk_bf8_f32 v1, v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f32 v1, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x01,0x40,0x36,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_pk_bf8_f32 v1, v1, v2, v3 op_sel:[0,0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f32 v1, v1, -v2, |v3| op_sel:[0,0,0,1] ; encoding: [0x01,0x44,0x36,0xd2,0x01,0x05,0x0e,0x44]
v_cvt_scalef32_pk_bf8_f32 v1, v1, -v2, |v3| op_sel:[0,0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f32 v1, v1, s2, 3 op_sel:[0,0,0,1] ; encoding: [0x01,0x40,0x36,0xd2,0x01,0x05,0x0c,0x02]
v_cvt_scalef32_pk_bf8_f32 v1, v1, s2, 3 op_sel:[0,0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp8 v[2:3], v2, v3 ; encoding: [0x02,0x00,0x39,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f32_fp8 v[2:3], v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp8 v[2:3], v2, s3 ; encoding: [0x02,0x00,0x39,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f32_fp8 v[2:3], v2, s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp8 v[2:3], s2, 3 ; encoding: [0x02,0x00,0x39,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f32_fp8 v[2:3], s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp8 v[2:3], v2, v3 op_sel:[1,0,0] ; encoding: [0x02,0x08,0x39,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f32_fp8 v[2:3], v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp8 v[2:3], v2, s3 op_sel:[1,0,0] ; encoding: [0x02,0x08,0x39,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f32_fp8 v[2:3], v2, s3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp8 v[2:3], s2, 3 op_sel:[1,0,0] ; encoding: [0x02,0x08,0x39,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f32_fp8 v[2:3], s2, 3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_bf8 v[2:3], v2, v3 ; encoding: [0x02,0x00,0x3a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f32_bf8 v[2:3], v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_bf8 v[2:3], v2, s3 ; encoding: [0x02,0x00,0x3a,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f32_bf8 v[2:3], v2, s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_bf8 v[2:3], s2, 3 ; encoding: [0x02,0x00,0x3a,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f32_bf8 v[2:3], s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_bf8 v[2:3], v2, v3 op_sel:[1,0,0] ; encoding: [0x02,0x08,0x3a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f32_bf8 v[2:3], v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_bf8 v[2:3], v2, s3 op_sel:[1,0,0] ; encoding: [0x02,0x08,0x3a,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f32_bf8 v[2:3], v2, s3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_bf8 v[2:3], s2, 3 op_sel:[1,0,0] ; encoding: [0x02,0x08,0x3a,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f32_bf8 v[2:3], s2, 3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f16 v1, v2, v3    ; encoding: [0x01,0x00,0x40,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp8_f16 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f16 v1, -v2, |v3| ; encoding: [0x01,0x02,0x40,0xd2,0x02,0x07,0x02,0x20]
v_cvt_scalef32_pk_fp8_f16 v1, -v2, |v3|

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f16 v1, s2, 3     ; encoding: [0x01,0x00,0x40,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_fp8_f16 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f16 v1, v2, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x40,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp8_f16 v1, v2, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f16 v1, -v2, |v3| op_sel:[0,0,1] ; encoding: [0x01,0x42,0x40,0xd2,0x02,0x07,0x02,0x20]
v_cvt_scalef32_pk_fp8_f16 v1, -v2, |v3| op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_f16 v1, s2, 3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x40,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_fp8_f16 v1, s2, 3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_bf16 v1, v2, v3   ; encoding: [0x01,0x00,0x44,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp8_bf16 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_bf16 v1, -v2, |v3| ; encoding: [0x01,0x02,0x44,0xd2,0x02,0x07,0x02,0x20]
v_cvt_scalef32_pk_fp8_bf16 v1, -v2, |v3|

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_bf16 v1, s2, 3    ; encoding: [0x01,0x00,0x44,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_fp8_bf16 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_bf16 v1, v2, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x44,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp8_bf16 v1, v2, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_bf16 v1, -v2, |v3| op_sel:[0,0,1] ; encoding: [0x01,0x42,0x44,0xd2,0x02,0x07,0x02,0x20]
v_cvt_scalef32_pk_fp8_bf16 v1, -v2, |v3| op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp8_bf16 v1, s2, 3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x44,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_fp8_bf16 v1, s2, 3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f16 v1, v2, v3    ; encoding: [0x01,0x00,0x41,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf8_f16 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f16 v1, -v2, |v3| ; encoding: [0x01,0x02,0x41,0xd2,0x02,0x07,0x02,0x20]
v_cvt_scalef32_pk_bf8_f16 v1, -v2, |v3|

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f16 v1, s2, 3     ; encoding: [0x01,0x00,0x41,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf8_f16 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f16 v1, v2, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x41,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf8_f16 v1, v2, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f16 v1, -v2, |v3| op_sel:[0,0,1] ; encoding: [0x01,0x42,0x41,0xd2,0x02,0x07,0x02,0x20]
v_cvt_scalef32_pk_bf8_f16 v1, -v2, |v3| op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_f16 v1, s2, 3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x41,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf8_f16 v1, s2, 3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_bf16 v1, v2, v3   ; encoding: [0x01,0x00,0x45,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf8_bf16 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_bf16 v1, -v2, |v3| ; encoding: [0x01,0x02,0x45,0xd2,0x02,0x07,0x02,0x20]
v_cvt_scalef32_pk_bf8_bf16 v1, -v2, |v3|

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_bf16 v1, s2, 3    ; encoding: [0x01,0x00,0x45,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf8_bf16 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_bf16 v1, v2, v3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x45,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf8_bf16 v1, v2, v3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_bf16 v1, -v2, |v3| op_sel:[0,0,1] ; encoding: [0x01,0x42,0x45,0xd2,0x02,0x07,0x02,0x20]
v_cvt_scalef32_pk_bf8_bf16 v1, -v2, |v3| op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf8_bf16 v1, s2, 3 op_sel:[0,0,1] ; encoding: [0x01,0x40,0x45,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf8_bf16 v1, s2, 3 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 ; encoding: [0x02,0x00,0x3f,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, s3 ; encoding: [0x02,0x00,0x3f,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], s2, 3 ; encoding: [0x02,0x00,0x3f,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 op_sel:[1,0,0] ; encoding: [0x02,0x08,0x3f,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, s3 op_sel:[1,0,0] ; encoding: [0x02,0x08,0x3f,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, s3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], s2, 3 op_sel:[1,0,0] ; encoding: [0x02,0x08,0x3f,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], s2, 3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 op_sel:[0,1,0] ; encoding: [0x02,0x10,0x3f,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, s3 op_sel:[0,1,0] ; encoding: [0x02,0x10,0x3f,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, s3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], s2, 3 op_sel:[0,1,0] ; encoding: [0x02,0x10,0x3f,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], s2, 3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 op_sel:[1,1,0] ; encoding: [0x02,0x18,0x3f,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, s3 op_sel:[1,1,0] ; encoding: [0x02,0x18,0x3f,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, s3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f32_fp4 v[2:3], s2, 3 op_sel:[1,1,0] ; encoding: [0x02,0x18,0x3f,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f32_fp4 v[2:3], s2, 3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 ; encoding: [0x01,0x00,0x3d,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, -v2, |v3| ; encoding: [0x01,0x04,0x3d,0xd2,0x01,0x05,0x0e,0x44]
v_cvt_scalef32_pk_fp4_f32 v1, v1, -v2, |v3|

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, s2, 3 ; encoding: [0x01,0x00,0x3d,0xd2,0x01,0x05,0x0c,0x02]
v_cvt_scalef32_pk_fp4_f32 v1, v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x01,0x20,0x3d,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 op_sel:[0,0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, -v2, |v3| op_sel:[0,0,1,0] ; encoding: [0x01,0x24,0x3d,0xd2,0x01,0x05,0x0e,0x44]
v_cvt_scalef32_pk_fp4_f32 v1, v1, -v2, |v3| op_sel:[0,0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, s2, 3 op_sel:[0,0,1,0] ; encoding: [0x01,0x20,0x3d,0xd2,0x01,0x05,0x0c,0x02]
v_cvt_scalef32_pk_fp4_f32 v1, v1, s2, 3 op_sel:[0,0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x01,0x40,0x3d,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 op_sel:[0,0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, -v2, |v3| op_sel:[0,0,0,1] ; encoding: [0x01,0x44,0x3d,0xd2,0x01,0x05,0x0e,0x44]
v_cvt_scalef32_pk_fp4_f32 v1, v1, -v2, |v3| op_sel:[0,0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, s2, 3 op_sel:[0,0,0,1] ; encoding: [0x01,0x40,0x3d,0xd2,0x01,0x05,0x0c,0x02]
v_cvt_scalef32_pk_fp4_f32 v1, v1, s2, 3 op_sel:[0,0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 op_sel:[0,0,1,1] ; encoding: [0x01,0x60,0x3d,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 op_sel:[0,0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, -v2, |v3| op_sel:[0,0,1,1] ; encoding: [0x01,0x64,0x3d,0xd2,0x01,0x05,0x0e,0x44]
v_cvt_scalef32_pk_fp4_f32 v1, v1, -v2, |v3| op_sel:[0,0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_fp4_f32 v1, v1, s2, 3 op_sel:[0,0,1,1] ; encoding: [0x01,0x60,0x3d,0xd2,0x01,0x05,0x0c,0x02]
v_cvt_scalef32_pk_fp4_f32 v1, v1, s2, 3 op_sel:[0,0,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, v2, v3    ; encoding: [0x01,0x00,0x50,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, v2, s3    ; encoding: [0x01,0x00,0x50,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, v2, s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, s2, 3     ; encoding: [0x01,0x00,0x50,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x50,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, v2, s3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x50,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, v2, s3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, s2, 3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x50,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, s2, 3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x50,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, v2, s3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x50,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, v2, s3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, s2, 3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x50,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, s2, 3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x50,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, v2, s3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x50,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, v2, s3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp4 v1, s2, 3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x50,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f16_fp4 v1, s2, 3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3   ; encoding: [0x01,0x00,0x51,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, v2, s3   ; encoding: [0x01,0x00,0x51,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, v2, s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, s2, 3    ; encoding: [0x01,0x00,0x51,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x51,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, v2, s3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x51,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, v2, s3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, s2, 3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x51,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, s2, 3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x51,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, v2, s3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x51,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, v2, s3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, s2, 3 op_sel:[0,1,0] ; encoding: [0x01,0x10,0x51,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, s2, 3 op_sel:[0,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x51,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, v2, s3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x51,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, v2, s3 op_sel:[1,1,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp4 v1, s2, 3 op_sel:[1,1,0] ; encoding: [0x01,0x18,0x51,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf16_fp4 v1, s2, 3 op_sel:[1,1,0]