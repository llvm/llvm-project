// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX13 --strict-whitespace --implicit-check-not=error %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX13 --strict-whitespace --implicit-check-not=error %s

v_dot2_bf16_bf16 v5, v1, v2, s3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_bf16_bf16_e64_dpp v0, s1, v2, v3 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_bf16_bf16_e64_dpp v0, v1, v2, v3 dpp8:[0,1,2,3,4,4,4,4]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f16_f16 v5, v1, v2, s3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f16_f16_e64_dpp v0, v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 fi:1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2_f16_f16_e64_dpp v0, v1, v2, v3 dpp8:[0,1,2,3,4,4,4,4]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_permlane16_b32 v5, v1, s2, s3 op_sel:[0, 0, 0, 1]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_permlanex16_b32 v5, v1, s2, s3 op_sel:[0, 0, 1, 0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_permlane16_var_b32 v5, v1, v2 clamp
// GFX13: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 clamp
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 div:2
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 div:2
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 mul:1
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 mul:1
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 -v5, v1, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 -v5, v1, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                     ^

v_permlane16_var_b32 v5, -v1, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, -v1, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                         ^

v_permlane16_var_b32 v5, v1, -v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, -v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                             ^

v_permlane16_var_b32 -|v5|, v1, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 -|v5|, v1, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                     ^

v_permlane16_var_b32 v5, -v1, |v2| op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, -v1, |v2| op_sel:[0, 1]
// GFX13-NEXT:{{^}}                         ^

v_permlane16_var_b32 v5, v1, -|v2| op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, -|v2| op_sel:[0, 1]
// GFX13-NEXT:{{^}}                             ^

v_permlane16_var_b32 |v5|, v1, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 |v5|, v1, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                     ^

v_permlane16_var_b32 v5, |v1|, v2 op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, |v1|, v2 op_sel:[0, 1]
// GFX13-NEXT:{{^}}                         ^

v_permlane16_var_b32 v5, v1, |v2| op_sel:[0, 1]
// GFX13: error: not a valid operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, |v2| op_sel:[0, 1]
// GFX13-NEXT:{{^}}                             ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[-1, 0]
// GFX13: error: invalid op_sel value
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[-1, 0]
// GFX13-NEXT:{{^}}                                        ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[1, -1]
// GFX13: error: invalid op_sel value
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[1, -1]
// GFX13-NEXT:{{^}}                                           ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, 1]
// GFX13: error: invalid op_sel operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, 1]
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, -1]
// GFX13: error: invalid op_sel value
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, -1]
// GFX13-NEXT:{{^}}                                                 ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1, 0]
// GFX13: error: invalid op_sel operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1, 0]
// GFX13-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, -1, 0]
// GFX13: error: invalid op_sel value
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, -1, 0]
// GFX13-NEXT:{{^}}                                              ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1]
// GFX13: error: invalid op_sel operand
// GFX13-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1]
// GFX13-NEXT:{{^}}                                ^
