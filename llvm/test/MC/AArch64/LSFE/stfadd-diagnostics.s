// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// STFADD
//------------------------------------------------------------------------------

stfadd h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfadd h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfadd s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfadd s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- stfaddl

stfaddl h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfaddl h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfaddl s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfaddl s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// STBFADD
//------------------------------------------------------------------------------

stbfadd s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfadd s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfadd d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfadd d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfadd h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfadd h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfadd h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfadd h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

// -- stbfaddl

stbfaddl s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfaddl s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfaddl d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfaddl d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfaddl h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfaddl h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfaddl h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfaddl h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}