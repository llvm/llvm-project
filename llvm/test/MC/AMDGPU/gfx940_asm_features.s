// RUN: llvm-mc -arch=amdgcn -mcpu=gfx940 -show-encoding %s | FileCheck --check-prefix=GFX940 --strict-whitespace %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefixes=NOT-GFX940,GFX90A --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=NOT-GFX940 --implicit-check-not=error: %s

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off sc0   ; encoding: [0x00,0x80,0x51,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off sc0

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off       ; encoding: [0x00,0x80,0x50,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off nosc0

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off sc1   ; encoding: [0x00,0x80,0x50,0xde,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off sc1

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off       ; encoding: [0x00,0x80,0x50,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off nosc1

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off nt    ; encoding: [0x00,0x80,0x52,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off nt

// NOT-GFX940: error: invalid operand for instruction
// GFX940: global_load_dword v2, v[2:3], off       ; encoding: [0x00,0x80,0x50,0xdc,0x02,0x00,0x7f,0x02]
global_load_dword v2, v[2:3], off nont

// GFX940: s_load_dword s2, s[2:3], 0x0 glc        ; encoding: [0x81,0x00,0x03,0xc0,0x00,0x00,0x00,0x00]
s_load_dword s2, s[2:3], 0x0 glc

// NOT-GFX940: error: invalid operand for instruction
// GFX940: buffer_load_dword v5, off, s[8:11], s3 sc0 nt sc1 ; encoding: [0x00,0xc0,0x52,0xe0,0x00,0x05,0x02,0x03]
buffer_load_dword v5, off, s[8:11], s3 sc0 nt sc1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_e32 v[2:3], v[4:5]            ; encoding: [0x04,0x71,0x04,0x7e]
v_mov_b64 v[2:3], v[4:5]

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_dpp v[2:3], v[4:5] row_newbcast:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x70,0x04,0x7e,0x04,0x51,0x01,0xff]
v_mov_b64 v[2:3], v[4:5]  row_newbcast:1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_e32 v[2:3], s[4:5]            ; encoding: [0x04,0x70,0x04,0x7e]
v_mov_b64 v[2:3], s[4:5]

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_e32 v[2:3], 1                 ; encoding: [0x81,0x70,0x04,0x7e]
v_mov_b64 v[2:3], 1

// NOT-GFX940: error: instruction not supported on this GPU
// GFX940: v_mov_b64_e32 v[2:3], 0x64              ; encoding: [0xff,0x70,0x04,0x7e,0x64,0x00,0x00,0x00]
v_mov_b64 v[2:3], 0x64

// NOT-GFX940: error: invalid operand for instruction
// GFX940: buffer_atomic_swap v5, off, s[8:11], s3 sc0 ; encoding: [0x00,0x40,0x00,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_swap v5, off, s[8:11], s3 sc0

// NOT-GFX940: error: invalid operand for instruction
// GFX940: buffer_atomic_swap v5, off, s[8:11], s3 nt ; encoding: [0x00,0x00,0x02,0xe1,0x00,0x05,0x02,0x03]
buffer_atomic_swap v5, off, s[8:11], s3 nt

// GFX90A: error: instruction not supported on this GPU
// GFX940: v_fmamk_f32 v0, v2, 0x42c80000, v3      ; encoding: [0x02,0x07,0x00,0x2e,0x00,0x00,0xc8,0x42]
v_fmamk_f32 v0, v2, 100.0, v3

// GFX90A: error: instruction not supported on this GPU
// GFX940: v_fmaak_f32 v0, v2, v3, 0x42c80000      ; encoding: [0x02,0x07,0x00,0x30,0x00,0x00,0xc8,0x42]
v_fmaak_f32 v0, v2, v3, 100.0
