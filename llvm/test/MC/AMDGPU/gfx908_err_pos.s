// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// inline constants are not allowed for this operand

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, 0
// CHECK: error: inline constants are not allowed for this operand
// CHECK-NEXT:{{^}}v_mfma_f32_32x32x1f32 a[0:31], v0, v1, 0
// CHECK-NEXT:{{^}}                                       ^

//==============================================================================
// invalid neg_hi value

v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[0,2]
// CHECK: error: invalid neg_hi value
// CHECK-NEXT:{{^}}v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[0,2]
// CHECK-NEXT:{{^}}                                        ^

//==============================================================================
// invalid neg_lo value

v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[2,0]
// CHECK: error: invalid neg_lo value
// CHECK-NEXT:{{^}}v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[2,0]
// CHECK-NEXT:{{^}}                                      ^

//==============================================================================
// invalid op_sel_hi value

v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[2,0]
// CHECK: error: invalid op_sel_hi value
// CHECK-NEXT:{{^}}v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[2,0]
// CHECK-NEXT:{{^}}                                         ^

//==============================================================================
// source operand must be either a VGPR or an inline constant

v_accvgpr_write a2, execz
// CHECK: error: source operand must be either a VGPR or an inline constant
// CHECK-NEXT:{{^}}v_accvgpr_write a2, execz
// CHECK-NEXT:{{^}}                    ^
