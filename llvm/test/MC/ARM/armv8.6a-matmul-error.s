// RUN: not llvm-mc -triple armv8a   -show-encoding -mattr=+i8mm < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple thumbv8a -show-encoding -mattr=+i8mm < %s 2>&1 | FileCheck %s


// VSMMLA, VUMMLA, VUSMMLA

// Data type specifier must match instruction

vsmmla.u8 q0, q1, q2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT:    vsmmla.u8 q0, q1, q2
// CHECK-NEXT: {{^      \^}}

vummla.s8 q0, q1, q2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT:    vummla.s8 q0, q1, q2
// CHECK-NEXT: {{^      \^}}

vusmmla.u8 q0, q1, q2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT:    vusmmla.u8 q0, q1, q2
// CHECK-NEXT: {{^       \^}}


// Incorrect register type

vsmmla.s8 d0, q1, q2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [q0, q15]
// CHECK-NEXT:    vsmmla.s8 d0, q1, q2
// CHECK-NEXT: {{^          \^}}

vummla.u8 q0, d1, q2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [q0, q15]
// CHECK-NEXT:    vummla.u8 q0, d1, q2
// CHECK-NEXT: {{^              \^}}

vusmmla.s8 q0, q1, d2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [q0, q15]
// CHECK-NEXT:    vusmmla.s8 q0, q1, d2
// CHECK-NEXT: {{^                   \^}}


// VUSDOT (vector)

// Data type specifier must match instruction

vusdot.u8 q0, q1, q2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT:    vusdot.u8 q0, q1, q2
// CHECK-NEXT: {{^      \^}}

// Mis-matched register types

vusdot.s8 q0, d1, d2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [d0, d31]
vusdot.s8 d0, q1, d2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [d0, d31]
vusdot.s8 d0, d1, q2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [d0, d31]


// VUSDOT, VSUDOT (by scalar)

// Data type specifier must match instruction

vusdot.u8 d0, d1, d2[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT:    vusdot.u8 d0, d1, d2[0]
// CHECK-NEXT: {{^      \^}}

vsudot.s8 d0, d1, d2[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT:    vsudot.s8 d0, d1, d2[0]
// CHECK-NEXT: {{^      \^}}

// Incorrect register types

vusdot.s8 q0, d1, d2[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid instruction, any one of the following would fix this:
// CHECK-NEXT: vusdot.s8 q0, d1, d2[0]
// CHECK: [[@LINE-3]]:{{[0-9]+}}: note: operand must be a register in range [d0, d31]
// CHECK-NEXT: vusdot.s8 q0, d1, d2[0]
// CHECK-NEXT: {{^       \^}}
// CHECK: [[@LINE-6]]:{{[0-9]+}}: note: operand must be a register in range [q0, q15]
// CHECK-NEXT: vusdot.s8 q0, d1, d2[0]
// CHECK-NEXT: {{^           \^}}

vusdot.s8 d0, q1, d2[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid instruction, any one of the following would fix this:
// CHECK-NEXT: vusdot.s8 d0, q1, d2[0]
// CHECK: [[@LINE-3]]:{{[0-9]+}}: note: operand must be a register in range [d0, d31]
// CHECK-NEXT: vusdot.s8 d0, q1, d2[0]
// CHECK-NEXT: {{^           \^}}
// CHECK: [[@LINE-6]]:{{[0-9]+}}: note: operand must be a register in range [q0, q15]
// CHECK-NEXT: vusdot.s8 d0, q1, d2[0]
// CHECK-NEXT: {{^       \^}}

vusdot.s8 q0, q1, q2[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid instruction, any one of the following would fix this:
// CHECK-NEXT: vusdot.s8 q0, q1, q2[0]
// CHECK: [[@LINE-3]]:{{[0-9]+}}: note: operand must be a register in range [d0, d15]
// CHECK-NEXT: vusdot.s8 q0, q1, q2[0]
// CHECK-NEXT: {{^               \^}}
// CHECK: [[@LINE-6]]:{{[0-9]+}}: note: too many operands for instruction
// CHECK-NEXT: vusdot.s8 q0, q1, q2[0]
// CHECK-NEXT: {{^                 \^}}

// Out of range lane index

vusdot.s8 d0, d1, d2[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
vsudot.u8 q0, q1, d2[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
