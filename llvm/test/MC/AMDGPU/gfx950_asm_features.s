// RUN: llvm-mc -triple=amdgcn -mcpu=gfx950 -show-encoding %s | FileCheck --check-prefix=GFX950 --strict-whitespace %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx942 %s 2>&1 | FileCheck --check-prefixes=NOT-GFX950 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefixes=NOT-GFX950 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=NOT-GFX950 --implicit-check-not=error: %s

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
// GFX950: v_permlane16_swap_b32_e32 v218, v219    ; encoding: [0xdb,0xb3,0xb4,0x7f]
v_permlane16_swap_b32 v218, v219

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e32 v1, v2        ; encoding: [0x02,0xb3,0x02,0x7e]
v_permlane16_swap_b32_e32 v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e32 v218, v219    ; encoding: [0xdb,0xb3,0xb4,0x7f]
v_permlane16_swap_b32_e32 v218, v219

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e64 v1, v2        ; encoding: [0x01,0x00,0x99,0xd1,0x02,0x01,0x00,0x00]
v_permlane16_swap_b32_e64 v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane16_swap_b32_e64 v218, v219    ; encoding: [0xda,0x00,0x99,0xd1,0xdb,0x01,0x00,0x00]
v_permlane16_swap_b32_e64 v218, v219

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
// GFX950: v_permlane32_swap_b32_e32 v218, v219    ; encoding: [0xdb,0xb5,0xb4,0x7f]
v_permlane32_swap_b32 v218, v219

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e32 v1, v2        ; encoding: [0x02,0xb5,0x02,0x7e]
v_permlane32_swap_b32_e32 v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e32 v218, v219    ; encoding: [0xdb,0xb5,0xb4,0x7f]
v_permlane32_swap_b32_e32 v218, v219

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e64 v1, v2        ; encoding: [0x01,0x00,0x9a,0xd1,0x02,0x01,0x00,0x00]
v_permlane32_swap_b32_e64 v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_permlane32_swap_b32_e64 v218, v219    ; encoding: [0xda,0x00,0x9a,0xd1,0xdb,0x01,0x00,0x00]
v_permlane32_swap_b32_e64 v218, v219

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

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_f32_fp6 v[2:33], v[2:7], v6 ; encoding: [0x02,0x00,0x56,0xd2,0x02,0x0d,0x02,0x00]
v_cvt_scalef32_pk32_f32_fp6 v[2:33], v[2:7], v6

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_f32_bf6 v[2:33], v[2:7], v6 ; encoding: [0x02,0x00,0x57,0xd2,0x02,0x0d,0x02,0x00]
v_cvt_scalef32_pk32_f32_bf6 v[2:33], v[2:7], v6

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_bf16_bf6 v[10:25], v[20:25], v8 ; encoding: [0x0a,0x00,0x63,0xd2,0x14,0x11,0x02,0x00]
v_cvt_scalef32_pk32_bf16_bf6 v[10:25], v[20:25], v8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_bf16_bf6 v[10:25], v[20:25], v8 ; encoding: [0x0a,0x00,0x63,0xd2,0x14,0x11,0x02,0x00]
v_cvt_scalef32_pk32_bf16_bf6 v[10:25], v[20:25], v8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_f16_bf6 v[10:25], v[20:25], v8 ; encoding: [0x0a,0x00,0x62,0xd2,0x14,0x11,0x02,0x00]
v_cvt_scalef32_pk32_f16_bf6 v[10:25], v[20:25], v8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_bf16_fp6 v[10:25], v[20:25], v8 ; encoding: [0x0a,0x00,0x61,0xd2,0x14,0x11,0x02,0x00]
v_cvt_scalef32_pk32_bf16_fp6 v[10:25], v[20:25], v8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_f16_fp6 v[10:25], v[20:25], v8 ; encoding: [0x0a,0x00,0x60,0xd2,0x14,0x11,0x02,0x00]
v_cvt_scalef32_pk32_f16_fp6 v[10:25], v[20:25], v8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_bf6_bf16 v[20:25], v[10:25], v8 ; encoding: [0x14,0x00,0x5b,0xd2,0x0a,0x11,0x02,0x00]
v_cvt_scalef32_pk32_bf6_bf16 v[20:25], v[10:25], v8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8 ; encoding: [0x14,0x00,0x5a,0xd2,0x0a,0x11,0x02,0x00]
v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8 ; encoding: [0x14,0x00,0x59,0xd2,0x0a,0x11,0x02,0x00]
v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 ; encoding: [0x14,0x00,0x58,0xd2,0x0a,0x11,0x02,0x00]
v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp8 v1, v2, v3    ; encoding: [0x01,0x00,0x48,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f16_fp8 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp8 v1, v2, s3    ; encoding: [0x01,0x00,0x48,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f16_fp8 v1, v2, s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp8 v1, s2, 3     ; encoding: [0x01,0x00,0x48,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f16_fp8 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp8 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x48,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f16_fp8 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp8 v1, v2, s3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x48,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f16_fp8 v1, v2, s3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_fp8 v1, s2, 3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x48,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f16_fp8 v1, s2, 3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_bf8 v1, v2, v3    ; encoding: [0x01,0x00,0x49,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f16_bf8 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_bf8 v1, v2, s3    ; encoding: [0x01,0x00,0x49,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f16_bf8 v1, v2, s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_bf8 v1, s2, 3     ; encoding: [0x01,0x00,0x49,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f16_bf8 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_bf8 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x49,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_f16_bf8 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_bf8 v1, v2, s3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x49,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_f16_bf8 v1, v2, s3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_f16_bf8 v1, s2, 3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x49,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_f16_bf8 v1, s2, 3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp8 v1, v2, v3   ; encoding: [0x01,0x00,0x69,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf16_fp8 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp8 v1, v2, s3   ; encoding: [0x01,0x00,0x69,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_bf16_fp8 v1, v2, s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp8 v1, s2, 3    ; encoding: [0x01,0x00,0x69,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf16_fp8 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp8 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x69,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf16_fp8 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp8 v1, v2, s3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x69,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_bf16_fp8 v1, v2, s3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_fp8 v1, s2, 3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x69,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf16_fp8 v1, s2, 3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_bf8 v1, v2, v3   ; encoding: [0x01,0x00,0x6a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf16_bf8 v1, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_bf8 v1, v2, s3   ; encoding: [0x01,0x00,0x6a,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_bf16_bf8 v1, v2, s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_bf8 v1, s2, 3    ; encoding: [0x01,0x00,0x6a,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf16_bf8 v1, s2, 3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_bf8 v1, v2, v3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x6a,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_bf16_bf8 v1, v2, v3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_bf8 v1, v2, s3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x6a,0xd2,0x02,0x07,0x00,0x00]
v_cvt_scalef32_pk_bf16_bf8 v1, v2, s3 op_sel:[1,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_pk_bf16_bf8 v1, s2, 3 op_sel:[1,0,0] ; encoding: [0x01,0x08,0x6a,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_bf16_bf8 v1, s2, 3 op_sel:[1,0,0]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_f16 v1, v2, v3    ; encoding: [0x01,0x00,0x4c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp4_f16 v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_f16 v1, s2, 3     ; encoding: [0x01,0x00,0x4c,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_fp4_f16 v1, s2, 3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_f16 v1, v2, v3 op_sel:[0,0,1,1] ; encoding: [0x01,0x60,0x4c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp4_f16 v1, v2, v3 op_sel:[0,0,1,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_f16 v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x01,0x40,0x4c,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp4_f16 v1, v2, v3 op_sel:[0,0,0,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_f16 v1, -|s2|, v3 ; encoding: [0x01,0x01,0x4c,0xd2,0x02,0x06,0x02,0x20]
v_cvt_scalef32_pk_fp4_f16 v1, -|s2|, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3   ; encoding: [0x01,0x00,0x4d,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_bf16 v1, s2, 3    ; encoding: [0x01,0x00,0x4d,0xd2,0x02,0x06,0x01,0x00]
v_cvt_scalef32_pk_fp4_bf16 v1, s2, 3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3 op_sel:[0,0,1,1] ; encoding: [0x01,0x60,0x4d,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3 op_sel:[0,0,1,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x01,0x40,0x4d,0xd2,0x02,0x07,0x02,0x00]
v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3 op_sel:[0,0,0,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_pk_fp4_bf16 v1, -|s2|, v3 ; encoding: [0x01,0x01,0x4d,0xd2,0x02,0x06,0x02,0x20]
v_cvt_scalef32_pk_fp4_bf16 v1, -|s2|, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], v6 ; encoding: [0x14,0x00,0x52,0xd2,0x0a,0x15,0x1a,0x04]
v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], v6

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], v6 ; encoding: [0x14,0x00,0x53,0xd2,0x0a,0x15,0x1a,0x04]
v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], v6

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], s6 ; encoding: [0x14,0x00,0x52,0xd2,0x0a,0x15,0x1a,0x00]
v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], s6

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], s6 ; encoding: [0x14,0x00,0x53,0xd2,0x0a,0x15,0x1a,0x00]
v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], s6

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], 22 ; encoding: [0x14,0x00,0x52,0xd2,0x0a,0x15,0x5a,0x02]
v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], 22

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], 11 ; encoding: [0x14,0x00,0x53,0xd2,0x0a,0x15,0x2e,0x02]
v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], 11

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0xff,0x02,0x03]
buffer_atomic_pk_add_bf16 v255, off, s[8:11], s3 offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x03,0x03]
buffer_atomic_pk_add_bf16 v5, off, s[12:15], s3 offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x18,0x03]
buffer_atomic_pk_add_bf16 v5, off, s[96:99], s3 offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x02,0x65]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], s101 offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x02,0x7c]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], m0 offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x48,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_pk_add_bf16 v5, v0, s[8:11], s3 idxen offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x48,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_pk_add_bf16 v5, v0, s[8:11], s3 offen offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x48,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x48,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x48,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:7

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x02,0x80]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], 0 offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x02,0xc1]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], -1 offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x02,0xf0]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], 0.5 offset:4095

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: buffer_atomic_pk_add_bf16 v5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x02,0xf7]
buffer_atomic_pk_add_bf16 v5, off, s[8:11], -4.0 offset:4095


// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_maximum3_f32 v1, v2, v3, v4           ; encoding: [0x01,0x00,0xa9,0xd2,0x02,0x07,0x12,0x04]
v_maximum3_f32 v1, v2, v3, v4

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_maximum3_f32 v1, -v2, -v3, -v4        ; encoding: [0x01,0x00,0xa9,0xd2,0x02,0x07,0x12,0xe4]
v_maximum3_f32 v1, -v2, -v3, -v4

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_maximum3_f32 v1, -|v2|, -|v3|, -|v4|  ; encoding: [0x01,0x07,0xa9,0xd2,0x02,0x07,0x12,0xe4]
v_maximum3_f32 v1, -|v2|, -|v3|, -|v4|

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_maximum3_f32 v1, 0, 1.0, v3           ; encoding: [0x01,0x00,0xa9,0xd2,0x80,0xe4,0x0d,0x04]
v_maximum3_f32 v1, 0.0, 1.0, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_maximum3_f32 v2, 0, v3, 1.0           ; encoding: [0x02,0x00,0xa9,0xd2,0x80,0x06,0xca,0x03]
v_maximum3_f32 v2, 0.0, v3, 1.0

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_maximum3_f32 v1, s8, v3, 1.0          ; encoding: [0x01,0x00,0xa9,0xd2,0x08,0x06,0xca,0x03]
v_maximum3_f32 v1, s8, v3, 1.0

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_maximum3_f32 v1, v2, s8, v3           ; encoding: [0x01,0x00,0xa9,0xd2,0x02,0x11,0x0c,0x04]
v_maximum3_f32 v1, v2, s8, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_minimum3_f32 v0, v1, v2, v3           ; encoding: [0x00,0x00,0xa8,0xd2,0x01,0x05,0x0e,0x04]
v_minimum3_f32 v0, v1, v2, v3


// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v1, v2, v3, v4        ; encoding: [0x01,0x40,0x9b,0xd3,0x02,0x07,0x12,0x1c]
v_pk_minimum3_f16 v1, v2, v3, v4

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v1, v2, v3, 2.0       ; encoding: [0x01,0x40,0x9b,0xd3,0x02,0x07,0xd2,0x1b]
v_pk_minimum3_f16 v1, v2, v3, 2.0

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v1, v2, 2.0, v3       ; encoding: [0x01,0x40,0x9b,0xd3,0x02,0xe9,0x0d,0x1c]
v_pk_minimum3_f16 v1, v2, 2.0, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v1, 2.0, v2, v3       ; encoding: [0x01,0x40,0x9b,0xd3,0xf4,0x04,0x0e,0x1c]
v_pk_minimum3_f16 v1, 2.0, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v1, v2, v3, v4 clamp  ; encoding: [0x01,0xc0,0x9b,0xd3,0x02,0x07,0x12,0x1c]
v_pk_minimum3_f16 v1, v2, v3, v4 clamp

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v8, v0, s8, v1        ; encoding: [0x08,0x40,0x9b,0xd3,0x00,0x11,0x04,0x1c]
v_pk_minimum3_f16 v8, v0, s8, v1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v8, v0, v1, s8        ; encoding: [0x08,0x40,0x9b,0xd3,0x00,0x03,0x22,0x18]
v_pk_minimum3_f16 v8, v0, v1, s8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v8, v0, s0, v1        ; encoding: [0x08,0x40,0x9b,0xd3,0x00,0x01,0x04,0x1c]
v_pk_minimum3_f16 v8, v0, s0, v1 neg_lo:[0,0,0] neg_hi:[0,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v8, v0, s0, v1        ; encoding: [0x08,0x40,0x9b,0xd3,0x00,0x01,0x04,0x1c]
v_pk_minimum3_f16 v8, v0, s0, v1 op_sel:[0,0,0] op_sel_hi:[1,1,1] neg_lo:[0,0,0] neg_hi:[0,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v8, v0, s0, v1        ; encoding: [0x08,0x40,0x9b,0xd3,0x00,0x01,0x04,0x1c]
v_pk_minimum3_f16 v8, v0, s0, v1 op_sel:[0,0,0] op_sel_hi:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v8, v0, s0, v1 op_sel_hi:[0,0,0] ; encoding: [0x08,0x00,0x9b,0xd3,0x00,0x01,0x04,0x04]
v_pk_minimum3_f16 v8, v0, s0, v1 op_sel:[0,0,0] op_sel_hi:[0,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_minimum3_f16 v8, v0, s0, v1 op_sel:[0,0,1] op_sel_hi:[0,0,1] ; encoding: [0x08,0x60,0x9b,0xd3,0x00,0x01,0x04,0x04]
v_pk_minimum3_f16 v8, v0, s0, v1 op_sel:[0,0,1] op_sel_hi:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v1, v2, v3, v4        ; encoding: [0x01,0x40,0x9c,0xd3,0x02,0x07,0x12,0x1c]
v_pk_maximum3_f16 v1, v2, v3, v4

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v1, v2, v3, 2.0       ; encoding: [0x01,0x40,0x9c,0xd3,0x02,0x07,0xd2,0x1b]
v_pk_maximum3_f16 v1, v2, v3, 2.0

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v1, v2, 2.0, v3       ; encoding: [0x01,0x40,0x9c,0xd3,0x02,0xe9,0x0d,0x1c]
v_pk_maximum3_f16 v1, v2, 2.0, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v1, 2.0, v2, v3       ; encoding: [0x01,0x40,0x9c,0xd3,0xf4,0x04,0x0e,0x1c]
v_pk_maximum3_f16 v1, 2.0, v2, v3

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v1, v2, v3, v4 clamp  ; encoding: [0x01,0xc0,0x9c,0xd3,0x02,0x07,0x12,0x1c]
v_pk_maximum3_f16 v1, v2, v3, v4 clamp

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v8, v0, s8, v1        ; encoding: [0x08,0x40,0x9c,0xd3,0x00,0x11,0x04,0x1c]
v_pk_maximum3_f16 v8, v0, s8, v1

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v8, v0, v1, s8        ; encoding: [0x08,0x40,0x9c,0xd3,0x00,0x03,0x22,0x18]
v_pk_maximum3_f16 v8, v0, v1, s8

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v8, v0, s0, v1        ; encoding: [0x08,0x40,0x9c,0xd3,0x00,0x01,0x04,0x1c]
v_pk_maximum3_f16 v8, v0, s0, v1 neg_lo:[0,0,0] neg_hi:[0,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v8, v0, s0, v1        ; encoding: [0x08,0x40,0x9c,0xd3,0x00,0x01,0x04,0x1c]
v_pk_maximum3_f16 v8, v0, s0, v1 op_sel:[0,0,0] op_sel_hi:[1,1,1] neg_lo:[0,0,0] neg_hi:[0,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v8, v0, s0, v1        ; encoding: [0x08,0x40,0x9c,0xd3,0x00,0x01,0x04,0x1c]
v_pk_maximum3_f16 v8, v0, s0, v1 op_sel:[0,0,0] op_sel_hi:[1,1,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v8, v0, s0, v1 op_sel_hi:[0,0,0] ; encoding: [0x08,0x00,0x9c,0xd3,0x00,0x01,0x04,0x04]
v_pk_maximum3_f16 v8, v0, s0, v1 op_sel:[0,0,0] op_sel_hi:[0,0,0]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_pk_maximum3_f16 v8, v0, s0, v1 op_sel:[0,0,1] op_sel_hi:[0,0,1] ; encoding: [0x08,0x60,0x9c,0xd3,0x00,0x01,0x04,0x04]
v_pk_maximum3_f16 v8, v0, s0, v1 op_sel:[0,0,1] op_sel_hi:[0,0,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, v5 ; encoding: [0x00,0x00,0x4f,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, v5 ; encoding: [0x00,0x00,0x4e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, v5 ; encoding: [0x00,0x00,0x3e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, v5 op_sel:[0,0,1,1] ; encoding: [0x00,0x60,0x4f,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, v5 op_sel:[0,0,1,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, v5 op_sel:[0,0,1,1] ; encoding: [0x00,0x60,0x4e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, v5 op_sel:[0,0,1,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, v5 op_sel:[0,0,1,1] ; encoding: [0x00,0x60,0x3e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, v5 op_sel:[0,0,1,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, v5 op_sel:[0,0,0,1] ; encoding: [0x00,0x40,0x4f,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, v5 op_sel:[0,0,0,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, v5 op_sel:[0,0,0,1] ; encoding: [0x00,0x40,0x4e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, v5 op_sel:[0,0,0,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, v5 op_sel:[0,0,0,1] ; encoding: [0x00,0x40,0x3e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, v5 op_sel:[0,0,0,1]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, v5 op_sel:[0,0,1,0] ; encoding: [0x00,0x20,0x4f,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, v5 op_sel:[0,0,1,0]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, v5 op_sel:[0,0,1,0] ; encoding: [0x00,0x20,0x4e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, v5 op_sel:[0,0,1,0]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, v5 op_sel:[0,0,1,0] ; encoding: [0x00,0x20,0x3e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, v5 op_sel:[0,0,1,0]

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_bf16 v0, -v2, v4, v5 ; encoding: [0x00,0x00,0x4f,0xd2,0x02,0x09,0x16,0x24]
v_cvt_scalef32_sr_pk_fp4_bf16 v0, -v2, v4, v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, -v5 ; encoding: [0x00,0x00,0x4f,0xd2,0x02,0x09,0x16,0x84]
v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, -v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, |v5| ; encoding: [0x00,0x04,0x4f,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_bf16 v0, v2, v4, |v5|

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_bf16 v0, |v2|, v4, v5 ; encoding: [0x00,0x01,0x4f,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_bf16 v0, |v2|, v4, v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f16 v0, -v2, v4, v5 ; encoding: [0x00,0x00,0x4e,0xd2,0x02,0x09,0x16,0x24]
v_cvt_scalef32_sr_pk_fp4_f16 v0, -v2, v4, v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, -v5 ; encoding: [0x00,0x00,0x4e,0xd2,0x02,0x09,0x16,0x84]
v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, -v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f16 v0, |v2|, v4, v5 ; encoding: [0x00,0x01,0x4e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f16 v0, |v2|, v4, v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, |v5| ; encoding: [0x00,0x04,0x4e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f16 v0, v2, v4, |v5|

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f32 v0, -v[2:3], v4, v5 ; encoding: [0x00,0x00,0x3e,0xd2,0x02,0x09,0x16,0x24]
v_cvt_scalef32_sr_pk_fp4_f32 v0, -v[2:3], v4, v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, -v5 ; encoding: [0x00,0x00,0x3e,0xd2,0x02,0x09,0x16,0x84]
v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, -v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f32 v0, |v[2:3]|, v4, v5 ; encoding: [0x00,0x01,0x3e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f32 v0, |v[2:3]|, v4, v5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, |v5| ; encoding: [0x00,0x04,0x3e,0xd2,0x02,0x09,0x16,0x04]
v_cvt_scalef32_sr_pk_fp4_f32 v0, v[2:3], v4, |v5|

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_sr_f16_f32 v0, v1, v2             ; encoding: [0x00,0x00,0xa6,0xd2,0x01,0x05,0x02,0x00]
v_cvt_sr_f16_f32 v0, v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_sr_bf16_f32 v0, v1, v2            ; encoding: [0x00,0x00,0xa7,0xd2,0x01,0x05,0x02,0x00]
v_cvt_sr_bf16_f32 v0, v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_sr_f16_f32 v0, v1, v2 op_sel:[0,0,1] ; encoding: [0x00,0x40,0xa6,0xd2,0x01,0x05,0x02,0x00]
v_cvt_sr_f16_f32 v0, v1, v2 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_sr_bf16_f32 v0, v1, v2 op_sel:[0,0,1] ; encoding: [0x00,0x40,0xa7,0xd2,0x01,0x05,0x02,0x00]
v_cvt_sr_bf16_f32 v0, v1, v2 op_sel:[0,0,1]

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_sr_f16_f32 v0, -v1, v2            ; encoding: [0x00,0x00,0xa6,0xd2,0x01,0x05,0x02,0x20]
v_cvt_sr_f16_f32 v0, -v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_sr_f16_f32 v0, |v1|, v2           ; encoding: [0x00,0x01,0xa6,0xd2,0x01,0x05,0x02,0x00]
v_cvt_sr_f16_f32 v0, |v1|, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_sr_bf16_f32 v0, -v1, v2           ; encoding: [0x00,0x00,0xa7,0xd2,0x01,0x05,0x02,0x20]
v_cvt_sr_bf16_f32 v0, -v1, v2

// NOT-GFX950: :[[@LINE+2]]:{{[0-9]+}}: error:
// GFX950: v_cvt_sr_bf16_f32 v0, |v1|, v2          ; encoding: [0x00,0x01,0xa7,0xd2,0x01,0x05,0x02,0x00]
v_cvt_sr_bf16_f32 v0, |v1|, v2

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_bf16 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x47,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_bf8_bf16 v0, v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_bf16 v0, -v1, v2, v3 ; encoding: [0x00,0x00,0x47,0xd2,0x01,0x05,0x0e,0x24]
v_cvt_scalef32_sr_bf8_bf16 v0, -v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_bf16 v0, v1, v2, -v3 ; encoding: [0x00,0x00,0x47,0xd2,0x01,0x05,0x0e,0x84]
v_cvt_scalef32_sr_bf8_bf16 v0, v1, v2, -v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_bf16 v0, |v1|, v2, v3 ; encoding: [0x00,0x01,0x47,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_bf8_bf16 v0, |v1|, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_bf16 v0, v1, v2, |v3| ; encoding: [0x00,0x04,0x47,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_bf8_bf16 v0, v1, v2, |v3|

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f16 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x43,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_bf8_f16 v0, v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f16 v0, -v1, v2, v3 ; encoding: [0x00,0x00,0x43,0xd2,0x01,0x05,0x0e,0x24]
v_cvt_scalef32_sr_bf8_f16 v0, -v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f16 v0, v1, v2, -v3 ; encoding: [0x00,0x00,0x43,0xd2,0x01,0x05,0x0e,0x84]
v_cvt_scalef32_sr_bf8_f16 v0, v1, v2, -v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f16 v0, |v1|, v2, v3 ; encoding: [0x00,0x01,0x43,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_bf8_f16 v0, |v1|, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f16 v0, v1, v2, |v3| ; encoding: [0x00,0x04,0x43,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_bf8_f16 v0, v1, v2, |v3|

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f32 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x38,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_bf8_f32 v0, v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f32 v0, -v1, v2, v3 ; encoding: [0x00,0x00,0x38,0xd2,0x01,0x05,0x0e,0x24]
v_cvt_scalef32_sr_bf8_f32 v0, -v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f32 v0, v1, v2, -v3 ; encoding: [0x00,0x00,0x38,0xd2,0x01,0x05,0x0e,0x84]
v_cvt_scalef32_sr_bf8_f32 v0, v1, v2, -v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f32 v0, |v1|, v2, v3 ; encoding: [0x00,0x01,0x38,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_bf8_f32 v0, |v1|, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_bf8_f32 v0, v1, v2, |v3| ; encoding: [0x00,0x04,0x38,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_bf8_f32 v0, v1, v2, |v3|

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_bf16 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x46,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_fp8_bf16 v0, v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_bf16 v0, -v1, v2, v3 ; encoding: [0x00,0x00,0x46,0xd2,0x01,0x05,0x0e,0x24]
v_cvt_scalef32_sr_fp8_bf16 v0, -v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_bf16 v0, v1, v2, -v3 ; encoding: [0x00,0x00,0x46,0xd2,0x01,0x05,0x0e,0x84]
v_cvt_scalef32_sr_fp8_bf16 v0, v1, v2, -v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_bf16 v0, |v1|, v2, v3 ; encoding: [0x00,0x01,0x46,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_fp8_bf16 v0, |v1|, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_bf16 v0, v1, v2, |v3| ; encoding: [0x00,0x04,0x46,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_fp8_bf16 v0, v1, v2, |v3|

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f16 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x42,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_fp8_f16 v0, v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f16 v0, -v1, v2, v3 ; encoding: [0x00,0x00,0x42,0xd2,0x01,0x05,0x0e,0x24]
v_cvt_scalef32_sr_fp8_f16 v0, -v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f16 v0, v1, v2, -v3 ; encoding: [0x00,0x00,0x42,0xd2,0x01,0x05,0x0e,0x84]
v_cvt_scalef32_sr_fp8_f16 v0, v1, v2, -v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f16 v0, |v1|, v2, v3 ; encoding: [0x00,0x01,0x42,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_fp8_f16 v0, |v1|, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f16 v0, v1, v2, |v3| ; encoding: [0x00,0x04,0x42,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_fp8_f16 v0, v1, v2, |v3|

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f32 v0, v1, v2, v3 ; encoding: [0x00,0x00,0x37,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_fp8_f32 v0, v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f32 v0, -v1, v2, v3 ; encoding: [0x00,0x00,0x37,0xd2,0x01,0x05,0x0e,0x24]
v_cvt_scalef32_sr_fp8_f32 v0, -v1, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f32 v0, v1, v2, -v3 ; encoding: [0x00,0x00,0x37,0xd2,0x01,0x05,0x0e,0x84]
v_cvt_scalef32_sr_fp8_f32 v0, v1, v2, -v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f32 v0, |v1|, v2, v3 ; encoding: [0x00,0x01,0x37,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_fp8_f32 v0, |v1|, v2, v3

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_fp8_f32 v0, v1, v2, |v3| ; encoding: [0x00,0x04,0x37,0xd2,0x01,0x05,0x0e,0x04]
v_cvt_scalef32_sr_fp8_f32 v0, v1, v2, |v3|

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk32_bf6_bf16 v[0:5], v[6:21], v22, v23 ; encoding: [0x00,0x00,0x5f,0xd2,0x06,0x2d,0x5e,0x04]
v_cvt_scalef32_sr_pk32_bf6_bf16 v[0:5], v[6:21], v22, v23

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk32_bf6_f16 v[0:5], v[6:21], v22, v23 ; encoding: [0x00,0x00,0x5e,0xd2,0x06,0x2d,0x5e,0x04]
v_cvt_scalef32_sr_pk32_bf6_f16 v[0:5], v[6:21], v22, v23

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk32_fp6_bf16 v[0:5], v[6:21], v22, v23 ; encoding: [0x00,0x00,0x5d,0xd2,0x06,0x2d,0x5e,0x04]
v_cvt_scalef32_sr_pk32_fp6_bf16 v[0:5], v[6:21], v22, v23

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk32_fp6_f16 v[0:5], v[6:21], v22, v23 ; encoding: [0x00,0x00,0x5c,0xd2,0x06,0x2d,0x5e,0x04]
v_cvt_scalef32_sr_pk32_fp6_f16 v[0:5], v[6:21], v22, v23

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk32_bf6_f32 v[0:5], v[6:37], v38, v39 ; encoding: [0x00,0x00,0x55,0xd2,0x06,0x4d,0x9e,0x04]
v_cvt_scalef32_sr_pk32_bf6_f32 v[0:5], v[6:37], v38, v39

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_scalef32_sr_pk32_fp6_f32 v[0:5], v[6:37], v38, v39 ; encoding: [0x00,0x00,0x54,0xd2,0x06,0x4d,0x9e,0x04]
v_cvt_scalef32_sr_pk32_fp6_f32 v[0:5], v[6:37], v38, v39

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_f16_f32 v5, v1, v2             ; encoding: [0x05,0x00,0x67,0xd2,0x01,0x05,0x02,0x00]
v_cvt_pk_f16_f32 v5, v1, v2

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_f16_f32 v5, v255, v255         ; encoding: [0x05,0x00,0x67,0xd2,0xff,0xff,0x03,0x00]
v_cvt_pk_f16_f32 v5, v255, v255

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_f16_f32 v5, m0, 0.5            ; encoding: [0x05,0x00,0x67,0xd2,0x7c,0xe0,0x01,0x00]
v_cvt_pk_f16_f32 v5, m0, 0.5

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_f16_f32 v5, exec_lo, -1        ; encoding: [0x05,0x00,0x67,0xd2,0x7e,0x82,0x01,0x00]
v_cvt_pk_f16_f32 v5, exec_lo, -1

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_f16_f32 v5, -1, exec_hi        ; encoding: [0x05,0x00,0x67,0xd2,0xc1,0xfe,0x00,0x00]
v_cvt_pk_f16_f32 v5, -1, exec_hi

// NOT-GFX950: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_f16_f32 v5, 0.5, m0 mul:2      ; encoding: [0x05,0x00,0x67,0xd2,0xf0,0xf8,0x00,0x08]
v_cvt_pk_f16_f32 v5, 0.5, m0 mul:2
