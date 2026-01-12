# RUN: not llvm-mc -triple riscv64 -mattr=+xsmtvdot < %s 2>&1 \
# RUN:     | FileCheck %s

# NoSlide
smt.vmadot   v1, v2, v4
// CHECK: :[[@LINE-1]]:14: error: invalid operand for instruction

smt.vmadot   v2, v2, v4
// CHECK: :[[@LINE-1]]:14: error: the destination vector register group cannot overlap the source vector register group

smt.vmadot   v4, v2, v4
// CHECK: :[[@LINE-1]]:14: error: the destination vector register group cannot overlap the source vector register group

smt.vmadot   v2, v3, v4
// CHECK: :[[@LINE-1]]:14: error: the destination vector register group cannot overlap the source vector register group

smt.vmadot   v2, v4, v3
// CHECK: :[[@LINE-1]]:14: error: the destination vector register group cannot overlap the source vector register group

# slide
smt.vmadot1   v1, v2, v2
// CHECK: :[[@LINE-1]]:15: error: invalid operand for instruction

smt.vmadot1   v2, v5, v4
// CHECK: :[[@LINE-1]]:19: error: invalid operand for instruction

smt.vmadot1   v2, v2, v4
// CHECK: :[[@LINE-1]]:15: error: the destination vector register group cannot overlap the source vector register group

smt.vmadot1   v2, v4, v2
// CHECK: :[[@LINE-1]]:15: error: the destination vector register group cannot overlap the source vector register group

smt.vmadot1   v0, v1, v2
// CHECK: :[[@LINE-1]]:19: error: invalid operand for instruction

smt.vmadot1   v0, v2, v1
// CHECK: :[[@LINE-1]]:15: error: the destination vector register group cannot overlap the source vector register group
