// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s

v_mov_b16 v0.l, s0.h
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mov_b16 v0.l, ttmp0.h
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mov_b16 v0.l, a0.h
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
