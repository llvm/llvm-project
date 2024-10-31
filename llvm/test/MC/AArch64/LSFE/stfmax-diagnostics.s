// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// STFMAX
//------------------------------------------------------------------------------

stfmax h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmax h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfmax s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmax s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- stfmaxl

stfmaxl h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmaxl h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfmaxl s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmaxl s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// STBFMAX
//------------------------------------------------------------------------------

stbfmax s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmax s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfmax d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmax d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmax h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmax h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmax h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmax h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

// -- stbfmaxl

stbfmaxl s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxl s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfmaxl d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxl d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmaxl h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxl h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmaxl h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxl h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}