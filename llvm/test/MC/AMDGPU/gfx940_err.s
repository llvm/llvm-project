// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck --check-prefix=GFX940 --implicit-check-not=error: %s

v_mac_f32 v0, v1, v2
// FIXME: error message is incorrect
// GFX940: error: operands are not valid for this GPU or mode

v_mad_f32 v0, v1, v2, v3
// GFX940: error: instruction not supported on this GPU

v_madak_f32 v0, v1, v2, 0
// GFX940: error: instruction not supported on this GPU

v_madmk_f32 v0, v1, 0, v2
// GFX940: error: instruction not supported on this GPU

v_mad_legacy_f32 v0, v1, v2, v3
// GFX940: error: instruction not supported on this GPU
