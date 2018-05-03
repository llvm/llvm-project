// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of upper bound [-256, 255].

str p0, [x0, #-257, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-256, 255].
// CHECK-NEXT: str p0, [x0, #-257, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

str p0, [x0, #256, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-256, 255].
// CHECK-NEXT: str p0, [x0, #256, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

str z0, [x0, #-257, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-256, 255].
// CHECK-NEXT: str z0, [x0, #-257, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

str z0, [x0, #256, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-256, 255].
// CHECK-NEXT: str z0, [x0, #256, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
