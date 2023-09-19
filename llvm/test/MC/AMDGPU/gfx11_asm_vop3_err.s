// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s

v_permlane16_b32 v5, v1, s2, s3 op_sel:[0, 0, 0, 1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_permlanex16_b32 v5, v1, s2, s3 op_sel:[0, 0, 1, 0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand
