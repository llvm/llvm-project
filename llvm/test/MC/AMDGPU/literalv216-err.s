// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=GFX9 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck -check-prefix=GFX10 --implicit-check-not=error: %s

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

v_pk_mad_i16 v5, 0x3c00, 0x4000, 0x4400
// GFX9: error: invalid literal operand
// GFX10: error: invalid literal operand

v_pk_mad_i16 v5, 0x3c00, 0x4000, 2
// GFX9: error: invalid literal operand
// GFX10: error: invalid literal operand

v_pk_mad_i16 v5, 0x3c00, 3, 2
// GFX9: error: invalid literal operand

v_pk_mad_i16 v5, 3, 0x3c00, 2
// GFX9: error: invalid literal operand

v_pk_mad_i16 v5, 3, 2, 0x3c00
// GFX9: error: invalid literal operand
