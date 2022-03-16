// RUN: llvm-mc -arch=amdgcn -mcpu=gfx940 -show-encoding %s | FileCheck -check-prefix=GFX940 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck -check-prefix=GFX90A %s

v_accvgpr_write_b32 a10, s20
// GFX940: v_accvgpr_write_b32 a10, s20    ; encoding: [0x0a,0x40,0xd9,0xd3,0x14,0x00,0x00,0x18]
// GFX90A: error: source operand must be either a VGPR or an inline constant
