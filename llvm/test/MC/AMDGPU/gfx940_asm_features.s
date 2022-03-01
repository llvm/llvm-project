// RUN: llvm-mc -arch=amdgcn -mcpu=gfx940 -show-encoding %s | FileCheck --check-prefix=GFX940 --strict-whitespace %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefixes=GFX90A --implicit-check-not=error: %s

// GFX90A: error: instruction not supported on this GPU
// GFX940: v_fmamk_f32 v0, v2, 0x42c80000, v3      ; encoding: [0x02,0x07,0x00,0x2e,0x00,0x00,0xc8,0x42]
v_fmamk_f32 v0, v2, 100.0, v3

// GFX90A: error: instruction not supported on this GPU
// GFX940: v_fmaak_f32 v0, v2, v3, 0x42c80000      ; encoding: [0x02,0x07,0x00,0x30,0x00,0x00,0xc8,0x42]
v_fmaak_f32 v0, v2, v3, 100.0
