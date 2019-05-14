// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s 2>&1 | FileCheck -check-prefix=GFX9 %s

v_pk_add_f16 v1, -17, v2
// GFX9: error: invalid literal operand

v_pk_add_f16 v1, 65, v2
// GFX9: error: invalid literal operand

v_pk_add_f16 v1, 64.0, v2
// GFX9: error: invalid literal operand

v_pk_add_f16 v1, -0.15915494, v2
// GFX9: error: invalid literal operand

v_pk_add_f16 v1, -0.0, v2
// GFX9: error: invalid literal operand

v_pk_add_f16 v1, -32768, v2
// GFX9: error: invalid literal operand

v_pk_add_f16 v1, 32767, v2
// GFX9: error: invalid literal operand

v_pk_add_f16 v1, 0xffffffffffff000f, v2
// GFX9: error: invalid literal operand

v_pk_add_f16 v1, 0x1000ffff, v2
// GFX9: error: invalid literal operand
