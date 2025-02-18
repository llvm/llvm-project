// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=-real-true16 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s

v_fmaak_f32 v0, 0xff32, v0, 0
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed

v_fmaak_f16 v0, 0xff32, v0, 0
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed

v_fmamk_f32 v0, 0xff32, 1, v0
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed

v_fmamk_f16 v0, 0xff32, 1, v0
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
