// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12 --strict-whitespace --implicit-check-not=error %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12 --strict-whitespace --implicit-check-not=error %s

v_permlane16_b32 v5, v1, s2, s3 op_sel:[0, 0, 0, 1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_permlanex16_b32 v5, v1, s2, s3 op_sel:[0, 0, 1, 0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_permlane16_var_b32 v5, v1, v2 clamp
// GFX12: error: invalid operand for instruction
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 clamp
// GFX12-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 div:2
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 div:2
// GFX12-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 mul:1
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 mul:1
// GFX12-NEXT:{{^}}                                ^

v_permlane16_var_b32 -v5, v1, v2 op_sel:[0, 1]
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 -v5, v1, v2 op_sel:[0, 1]
// GFX12-NEXT:{{^}}                     ^

v_permlane16_var_b32 v5, -v1, v2 op_sel:[0, 1]
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, -v1, v2 op_sel:[0, 1]
// GFX12-NEXT:{{^}}                         ^

v_permlane16_var_b32 v5, v1, -v2 op_sel:[0, 1]
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, -v2 op_sel:[0, 1]
// GFX12-NEXT:{{^}}                             ^

v_permlane16_var_b32 -|v5|, v1, v2 op_sel:[0, 1]
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 -|v5|, v1, v2 op_sel:[0, 1]
// GFX12-NEXT:{{^}}                     ^

v_permlane16_var_b32 v5, -v1, |v2| op_sel:[0, 1]
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, -v1, |v2| op_sel:[0, 1]
// GFX12-NEXT:{{^}}                         ^

v_permlane16_var_b32 v5, v1, -|v2| op_sel:[0, 1]
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, -|v2| op_sel:[0, 1]
// GFX12-NEXT:{{^}}                             ^

v_permlane16_var_b32 |v5|, v1, v2 op_sel:[0, 1]
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 |v5|, v1, v2 op_sel:[0, 1]
// GFX12-NEXT:{{^}}                     ^

v_permlane16_var_b32 v5, |v1|, v2 op_sel:[0, 1]
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, |v1|, v2 op_sel:[0, 1]
// GFX12-NEXT:{{^}}                         ^

v_permlane16_var_b32 v5, v1, |v2| op_sel:[0, 1]
// GFX12: error: not a valid operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, |v2| op_sel:[0, 1]
// GFX12-NEXT:{{^}}                             ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[-1, 0]
// GFX12: error: invalid op_sel value
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[-1, 0]
// GFX12-NEXT:{{^}}                                        ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[1, -1]
// GFX12: error: invalid op_sel value
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[1, -1]
// GFX12-NEXT:{{^}}                                           ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, 1]
// GFX12: error: invalid op_sel operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, 1]
// GFX12-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, -1]
// GFX12: error: invalid op_sel value
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 0, -1]
// GFX12-NEXT:{{^}}                                                 ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1, 0]
// GFX12: error: invalid op_sel operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1, 0]
// GFX12-NEXT:{{^}}                                ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, -1, 0]
// GFX12: error: invalid op_sel value
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, -1, 0]
// GFX12-NEXT:{{^}}                                              ^

v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1]
// GFX12: error: invalid op_sel operand
// GFX12-NEXT:{{^}}v_permlane16_var_b32 v5, v1, v2 op_sel:[0, 0, 1]
// GFX12-NEXT:{{^}}                                ^

v_cvt_sr_bf8_f32 v1, v2, v3 byte_sel:4
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid byte_sel value.
// GFX12-NEXT:{{^}}v_cvt_sr_bf8_f32 v1, v2, v3 byte_sel:4
// GFX12-NEXT:{{^}}                            ^
