// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// STFMAXNM
//------------------------------------------------------------------------------

stfmaxnm h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmaxnm h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfmaxnm s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmaxnm s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- stfmaxnml

stfmaxnml h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmaxnml h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfmaxnml s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmaxnml s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// STBFMAXNM
//------------------------------------------------------------------------------

stbfmaxnm s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxnm s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfmaxnm d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxnm d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmaxnm h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxnm h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmaxnm h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxnm h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

// --stbfmaxnml

stbfmaxnml s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxnml s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfmaxnml d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxnml d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmaxnml h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxnml h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmaxnml h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmaxnml h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}