// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=GFX9ERR --implicit-check-not=error: %s

v_cvt_f16_u16_e64 v5, 0.5
// GFX9ERR: error: invalid literal operand

v_cvt_f16_u16_e64 v5, -4.0
// GFX9ERR: error: invalid literal operand

v_add_u16_e64 v5, v1, 0.5
// GFX9ERR: error: invalid literal operand

v_add_u16_e64 v5, v1, -4.0
// GFX9ERR: error: invalid literal operand

v_cvt_f16_i16_e64 v5, 0.5
// GFX9ERR: error: invalid literal operand

v_cvt_f16_i16_e64 v5, -4.0
// GFX9ERR: error: invalid literal operand

v_add_u16_e64 v5, 0.5, v2
// GFX9ERR: error: invalid literal operand

v_add_u16_e64 v5, -4.0, v2
// GFX9ERR: error: invalid literal operand

v_subrev_u16_e64 v5, v1, 0.5
// GFX9ERR: error: invalid literal operand

v_subrev_u16_e64 v5, v1, -4.0
// GFX9ERR: error: invalid literal operand
