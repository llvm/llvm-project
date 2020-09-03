// Tests for transactional memory extension instructions
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+tme < %s 2>&1   | FileCheck %s

tstart
// CHECK: error: too few operands for instruction
// CHECK-NEXT: tstart
tstart  x4, x5
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: tstart x4, x5
tstart  x4, #1
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: tstart x4, #1
tstart  sp
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: tstart sp

ttest
// CHECK: error: too few operands for instruction
// CHECK-NEXT: ttest
ttest  x4, x5
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: ttest x4, x5
ttest  x4, #1
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: ttest x4, #1
ttest  sp
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: ttest sp

tcommit  x4
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: tcommit x4
tcommit  sp
// CHECK: error: invalid operand for instruction
// CHECK-NEXT: tcommit sp


tcancel
// CHECK: error: too few operands for instruction
// CHECK-NEXT: tcancel
tcancel x0
// CHECK: error: immediate must be an integer in range [0, 65535]
// CHECK-NEXT: tcancel
tcancel #65536
// CHECK: error: immediate must be an integer in range [0, 65535]
// CHECK-NEXT: tcancel #65536

