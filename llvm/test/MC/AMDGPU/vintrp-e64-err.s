// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --implicit-check-not=error:
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck %s --implicit-check-not=error:

v_interp_p1_f32_e64 v5, 0.5, attr0.w
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1_f32_e64 v5, s1, attr0.w
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1ll_f16 v5, 0.5, attr0.w
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1ll_f16 v5, s1, attr0.w
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1lv_f16 v5, 0.5, attr0.w, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1lv_f16 v5, s1, attr0.w, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1lv_f16 v5, v1, attr31.w, 0.5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1lv_f16 v5, v1, attr31.w, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f16 v5, 0.5, attr0.w, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f16 v5, s1, attr0.w, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f16 v5, v1, attr1.w, 0.5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f16 v5, v1, attr1.w, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f32_e64 v5, 0.5, attr31.w
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f32_e64 v5, s1, attr31.w
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
