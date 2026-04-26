// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

ldclrpl  x22, xzr, [sp]
// CHECK-ERROR: error: invalid operand for instruction

ldclrpl  xzr, x22, [sp]
// CHECK-ERROR: error: invalid operand for instruction

ldsetpl  x22, xzr, [sp]
// CHECK-ERROR: error: invalid operand for instruction

ldsetpl  xzr, x22, [sp]
// CHECK-ERROR: error: invalid operand for instruction

swppl    x22, xzr, [sp]
// CHECK-ERROR: error: invalid operand for instruction

swppl    xzr, x22, [sp]
// CHECK-ERROR: error: invalid operand for instruction
