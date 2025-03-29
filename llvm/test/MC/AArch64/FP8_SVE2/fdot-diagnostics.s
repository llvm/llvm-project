// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+ssve-fp8dot2,+ssve-fp8dot4 \
// RUN: 2>&1 < %s | FileCheck %s

// FDOT2
// --------------------------------------------------------------------------//

// z register out of range for index

fdot z0.h, z0.b, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot z0.h, z0.b, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// Invalid vector lane index

fdot z0.h, z0.b, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fdot z0.h, z0.b, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot z0.h, z0.b, z0.b[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fdot z0.h, z0.b, z0.b[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid vector suffix

fdot z0.d, z0.b, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fdot z0.d, z0.b, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot z0.h, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fdot z0.h, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:



// FDOT4
// --------------------------------------------------------------------------//
// Invalid vector lane index

fdot z0.s, z0.b, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot z0.s, z0.b, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot z0.s, z0.b, z0.b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot z0.s, z0.b, z0.b[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid vector suffix

fdot z0.s, z0.s, z0.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fdot z0.s, z0.s, z0.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot z0.b, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fdot z0.b, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
