// RUN: not llvm-mc -triple=aarch64 -filetype=obj -show-encoding -mattr=+cmpbr 2>&1 < %s | FileCheck %s

//------------------------------------------------------------------------------
// Incorrect label

// -- cbgt

cbgt x5, x5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbgt x5, x5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbgt w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbgt w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbgt w5, #20, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbgt w5, #20, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbgt x5, #20, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbgt x5, #20, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cblt

cblt x5, #20, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cblt x5, #20, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cblt w5, #20, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cblt w5, #20, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbge

cbge x5, x5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbge x5, x5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbge w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbge w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhi

cbhi x2, x2, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhi x2, x2, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhi w2, w2, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhi w2, w2, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhi w2, #20, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhi w2, #20, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhi x2, #20, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhi x2, #20, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cblo

cblo w5, #20, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cblo w5, #20, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cblo x5, #20, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cblo x5, #20, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhs

cbhs x2, x2, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhs x2, x2, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhs w2, w2, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhs w2, w2, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbeq

cbeq x5, x5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbeq x5, x5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}

cbeq w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbeq w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}

cbeq w5, #20, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbeq w5, #20, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}

cbeq x5, #20, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbeq x5, #20, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}

// -- cbne

cbne w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbne w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}

cbne x5, x5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbne x5, x5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}

cbne x5, #20, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbne x5, #20, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}

cbne w5, #20, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbne w5, #20, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}

// -- cbhgt

cbhgt w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhgt w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhgt w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhgt w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhge

cbhge w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhge w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhge w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhge w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhhi

cbhhi w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhhi w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhhi w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhhi w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhhs

cbhhs w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhhs w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhhs w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhhs w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbheq

cbheq w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbheq w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbheq w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbheq w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhne

cbhne w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhne w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhne w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbhne w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbgt

cbbgt w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbgt w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbgt w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbgt w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbge

cbbge w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbge w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbge w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbge w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbhi

cbbhi w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbhi w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbhi w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbhi w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbhs

cbbhs w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbhs w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbhs w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbhs w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbeq

cbbeq w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbeq w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbeq w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbeq w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbne

cbbne w5, w5, #-1025
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbne w5, w5, #-1025
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbne w5, w5, #1021
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected label or encodable integer pc offset
// CHECK-NEXT: cbbne w5, w5, #1021
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:


//------------------------------------------------------------------------------
// Incorrect Operands

// -- cbhgt

cbhgt w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhgt w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhgt x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhgt x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhge

cbhge w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhge w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhge x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhge x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhhi

cbhhi w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhhi w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhhi x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhhi x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhhs

cbhhs w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhhs w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhhs x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhhs x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbneq

cbheq w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbheq w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbheq x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbheq x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhne

cbhne w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhne w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhne x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbhne x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbgt

cbbgt w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbgt w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbgt x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbgt x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbge

cbbge w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbge w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbge x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbge x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbhi

cbbhi w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbhi w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbhi x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbhi x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbhs

cbbhs w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbhs w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbhs x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbhs x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbeq

cbbeq w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbeq w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbeq x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cbbeq x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbbne

cbbne w5, #20, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bbne w5, #20, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbbne x5, x5, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bbne x5, x5, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// (Immediate) compare value out-of-range

// -- cbgt

cbgt w5, #-1, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cbgt w5, #-1, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbgt w5, #64, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cbgt w5, #64, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cblt

cblt w5, #-1, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cblt w5, #-1, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cblt w5, #64, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cblt w5, #64, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbhi

cbhi w5, #-1, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cbhi w5, #-1, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbhi w5, #64, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cbhi w5, #64, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cblo

cblo w5, #-1, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cblo w5, #-1, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cblo w5, #64, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cblo w5, #64, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbeq

cbeq w5, #-1, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cbeq w5, #-1, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbeq x5, #64, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cbeq x5, #64, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// -- cbne

cbne x5, #-1, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cbne x5, #-1, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

cbne w5, #64, #1020
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: cbne w5, #64, #1020
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}: