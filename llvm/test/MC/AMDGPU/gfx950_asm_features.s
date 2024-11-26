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
