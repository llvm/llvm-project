// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Generic checks
//===----------------------------------------------------------------------===//

v_mul_i32_i24 v1, v2, 100
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

//===----------------------------------------------------------------------===//
// _e32 checks
//===----------------------------------------------------------------------===//

// Immediate src1
v_mul_i32_i24_e32 v1, v2, 100
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

// sgpr src1
v_mul_i32_i24_e32 v1, v2, s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cndmask_b32_e32 v1, v2, v3, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

//===----------------------------------------------------------------------===//
// _e64 checks
//===----------------------------------------------------------------------===//

// Immediate src0
v_mul_i32_i24_e64 v1, 100, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

// Immediate src1
v_mul_i32_i24_e64 v1, v2, 100
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_add_i32_e32 v1, s[0:1], v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32_e32 v1, vcc, v2, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32_e32 v1, s[0:1], v2, v3, s[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32_e32 v1, vcc, v2, v3, -1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32_e32 v1, vcc, v2, v3, 123
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32_e32 v1, vcc, v2, v3, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32_e32 v1, -1, v2, v3, s0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32 v1, -1, v2, v3, vcc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32 v1, vcc, v2, v3, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32_e64 v1, s[0:1], v2, v3, 123
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32_e64 v1, 0, v2, v3, s[0:1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32_e64 v1, s[0:1], v2, v3, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_addc_u32 v1, s[0:1], v2, v3, 123
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

// TODO: Constant bus restrictions
