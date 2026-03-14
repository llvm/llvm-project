// RUN: not llvm-mc -triple arm -mattr=+v8.4a -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple thumb -mattr=+v8.4a -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

tsb
tsb 0
tsb #0
tsb foo

//CHECK-ERROR: error: too few operands for instruction
//CHECK-ERROR: tsb
//CHECK-ERROR: ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR: tsb 0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR: tsb #0
//CHECK-ERROR:     ^
//CHECK-ERROR: error: invalid operand for instruction
//CHECK-ERROR: tsb foo
//CHECK-ERROR:     ^
