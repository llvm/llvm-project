// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,+real-true16 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX11 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64,+real-true16 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX11 %s


// op_sel hi on vgpr but not on op_sel operand
v_add_f16 v0.h, v1.h, v2.h op_sel:[0,1,1]
// GFX11: op_sel operand conflicts with 16-bit operand suffix

// op_sel hi on op_sel operand but not on vgpr
v_add_f16 v0.h, v1.l, v2.h op_sel:[1,1,1]
// GFX11: op_sel operand conflicts with 16-bit operand suffix
