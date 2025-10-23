// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+mops-go,+mte < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// All operand must be different from each other

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setgop [x0]!, x0!, x1
setgop [x0]!, x1!, x0
setgop [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setgom [x0]!, x0!, x1
setgom [x0]!, x1!, x0
setgom [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setgoe [x0]!, x0!, x1
setgoe [x0]!, x1!, x0
setgoe [x1]!, x0!, x0

// SP cannot be used as argument at any position

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setgop [sp]!, x1!, x2
setgop [x0]!, sp!, x2
setgop [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setgom [sp]!, x1!, x2
setgom [x0]!, sp!, x2
setgom [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setgoe [sp]!, x1!, x2
setgoe [x0]!, sp!, x2
setgoe [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
setgop [xzr]!, x1!, x2

// CHECK-ERROR: error: invalid operand for instruction
setgom [xzr]!, x1!, x2

// CHECK-ERROR: error: invalid operand for instruction
setgoe [xzr]!, x1!, x2
