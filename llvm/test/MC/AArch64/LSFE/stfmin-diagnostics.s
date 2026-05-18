// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsfe 2>&1 < %s| FileCheck %s

//------------------------------------------------------------------------------
// STFMIN
//------------------------------------------------------------------------------

stfmin h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmin h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfmin s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfmin s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// -- stfminl

stfminl h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfminl h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stfminl s0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stfminl s0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//------------------------------------------------------------------------------
// STBFMIN
//------------------------------------------------------------------------------

stbfmin s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmin s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfmin d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmin d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmin h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmin h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfmin h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfmin h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

// -- stbfminl

stbfminl s0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminl s0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stbfminl d0, [x2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminl d0, [x2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfminl h0, [w2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminl h0, [w2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

stbfminl h0, [x2, #4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stbfminl h0, [x2, #4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}