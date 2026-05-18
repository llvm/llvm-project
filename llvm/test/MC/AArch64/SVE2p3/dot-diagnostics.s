// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element width

sdot z0.b, z0.b, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.b, z0.b, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.h, z0.h, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.h, z0.h, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.s, z0.s, z0.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.s, z0.s, z0.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.d, z0.d, z0.d[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.d, z0.d, z0.d[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.d, z0.s, z0.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.d, z0.s, z0.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.b, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.b, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.h, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.h, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.s, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.s, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.d, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.d, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.d, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sdot z0.d, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.b, z0.b, z0.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.b, z0.b, z0.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.h, z0.h, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.h, z0.h, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.s, z0.s, z0.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.s, z0.s, z0.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.d, z0.d, z0.d[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.d, z0.d, z0.d[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.d, z0.s, z0.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.d, z0.s, z0.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.b, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.b, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.h, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.h, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.s, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.s, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.d, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.d, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.d, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: udot z0.d, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid register range and index

sdot z0.h, z0.b, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sdot z0.h, z0.b, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.h, z0.b, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: sdot z0.h, z0.b, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sdot z0.h, z0.b, z0.b[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: sdot z0.h, z0.b, z0.b[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.h, z0.b, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: udot z0.h, z0.b, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.h, z0.b, z0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: udot z0.h, z0.b, z0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

udot z0.h, z0.b, z0.b[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: udot z0.h, z0.b, z0.b[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
