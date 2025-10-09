// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+mops,+mte < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.8a,+mte < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR


// All operand must be different from each other

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpyfp [x0]!, [x0]!, x1!
cpyfp [x0]!, [x1]!, x0!
cpyfp [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpyfm [x0]!, [x0]!, x1!
cpyfm [x0]!, [x1]!, x0!
cpyfm [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpyfe [x0]!, [x0]!, x1!
cpyfe [x0]!, [x1]!, x0!
cpyfe [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpyp [x0]!, [x0]!, x1!
cpyp [x0]!, [x1]!, x0!
cpyp [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpym [x0]!, [x0]!, x1!
cpym [x0]!, [x1]!, x0!
cpym [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid CPY instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid CPY instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid CPY instruction, source and size registers are the same
cpye [x0]!, [x0]!, x1!
cpye [x0]!, [x1]!, x0!
cpye [x1]!, [x0]!, x0!

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setp [x0]!, x0!, x1
setp [x0]!, x1!, x0
setp [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setm [x0]!, x0!, x1
setm [x0]!, x1!, x0
setm [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
sete [x0]!, x0!, x1
sete [x0]!, x1!, x0
sete [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setgp [x0]!, x0!, x1
setgp [x0]!, x1!, x0
setgp [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setgm [x0]!, x0!, x1
setgm [x0]!, x1!, x0
setgm [x1]!, x0!, x0

// CHECK-ERROR: error: invalid SET instruction, destination and size registers are the same
// CHECK-ERROR: error: invalid SET instruction, destination and source registers are the same
// CHECK-ERROR: error: invalid SET instruction, source and size registers are the same
setge [x0]!, x0!, x1
setge [x0]!, x1!, x0
setge [x1]!, x0!, x0

// SP cannot be used as argument at any position

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyfp [sp]!, [x1]!, x2!
cpyfp [x0]!, [sp]!, x2!
cpyfp [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyfm [sp]!, [x1]!, x2!
cpyfm [x0]!, [sp]!, x2!
cpyfm [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyfe [sp]!, [x1]!, x2!
cpyfe [x0]!, [sp]!, x2!
cpyfe [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyp [sp]!, [x2]!, x2!
cpyp [x0]!, [sp]!, x2!
cpyp [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpym [sp]!, [x2]!, x2!
cpym [x0]!, [sp]!, x2!
cpym [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpye [sp]!, [x2]!, x2!
cpye [x0]!, [sp]!, x2!
cpye [x0]!, [x1]!, sp!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setp [sp]!, x1!, x2
setp [x0]!, sp!, x2
setp [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setm [sp]!, x1!, x2
setm [x0]!, sp!, x2
setm [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
sete [sp]!, x1!, x2
sete [x0]!, sp!, x2
sete [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setgp [sp]!, x1!, x2
setgp [x0]!, sp!, x2
setgp [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setgm [sp]!, x1!, x2
setgm [x0]!, sp!, x2
setgm [x0]!, x1!, sp

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
setge [sp]!, x1!, x2
setge [x0]!, sp!, x2
setge [x0]!, x1!, sp

// XZR can only be used at:
//  - the size operand in CPY.
//  - the size or source operands in SET.

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyfp [xzr]!, [x1]!, x2!
cpyfp [x0]!, [xzr]!, x2!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyfm [xzr]!, [x1]!, x2!
cpyfm [x0]!, [xzr]!, x2!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyfe [xzr]!, [x1]!, x2!
cpyfe [x0]!, [xzr]!, x2!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpyp [xzr]!, [x2]!, x2!
cpyp [x0]!, [xzr]!, x2!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpym [xzr]!, [x2]!, x2!
cpym [x0]!, [xzr]!, x2!

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
cpye [xzr]!, [x2]!, x2!
cpye [x0]!, [xzr]!, x2!

// CHECK-ERROR: error: invalid operand for instruction
setp [xzr]!, x1!, x2

// CHECK-ERROR: error: invalid operand for instruction
setm [xzr]!, x1!, x2

// CHECK-ERROR: error: invalid operand for instruction
sete [xzr]!, x1!, x2

// CHECK-ERROR: error: invalid operand for instruction
setgp [xzr]!, x1!, x2

// CHECK-ERROR: error: invalid operand for instruction
setgm [xzr]!, x1!, x2

// CHECK-ERROR: error: invalid operand for instruction
setge [xzr]!, x1!, x2
