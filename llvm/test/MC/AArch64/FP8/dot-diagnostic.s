// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+fp8dot2,+fp8dot4 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Element size extension incorrect

fdot  v31.4h, v0.8h, v0.8b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot  v31.4h, v0.8h, v0.8b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot  v31.8h, v0.16b, v31.16h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fdot  v31.8h, v0.16b, v31.16h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot  v0.2s, v0.8s, v31.8b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fdot  v0.2s, v0.8s, v31.8b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot  v31.4s, v0, v31.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot  v31.4s, v0, v31.16b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

//--------------------------------------------------------------------------//
// Last Register range is between 0-15

fdot  v31.4h, v31.8b, v16.2b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot  v31.4h, v31.8b, v16.2b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot  v0.8h, v0.16b, v16.2b[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fdot  v0.8h, v0.16b, v16.2b[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Out of range index
fdot  v31.4h, v31.8b, v15.2b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fdot  v31.4h, v31.8b, v15.2b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot  v0.8h, v0.16b, v15.2b[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fdot  v0.8h, v0.16b, v15.2b[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot  v0.2s, v0.8b, v31.4b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot  v0.2s, v0.8b, v31.4b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdot  v0.4s, v31.16b, v0.4b[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fdot  v0.4s, v31.16b, v0.4b[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
