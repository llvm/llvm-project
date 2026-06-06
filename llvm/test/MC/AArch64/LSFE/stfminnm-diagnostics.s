// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// STFMINNM
//------------------------------------------------------------------------------

stfminnm h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfminnm h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfminnm s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfminnm s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- stfminnml

stfminnml h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfminnml h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfminnml s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfminnml s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// STBFMINNM
//------------------------------------------------------------------------------

stbfminnm s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminnm s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfminnm d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminnm d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfminnm h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminnm h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfminnm h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminnm h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

// -- stbfminnml

stbfminnml s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminnml s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfminnml d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminnml d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfminnml h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminnml h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfminnml h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminnml h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}