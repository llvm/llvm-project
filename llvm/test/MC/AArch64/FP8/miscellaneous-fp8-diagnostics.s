// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+fp8 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Element size extension incorrect

bf1cvtl v0.8h, v0.8h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bf1cvtl v0.8h, v0.8h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf1cvtl2 v0.8h, v0.16h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: bf1cvtl2 v0.8h, v0.16h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf2cvtl v0.8h, v0.8h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bf2cvtl v0.8h, v0.8h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf2cvtl2 v0.8h, v0.16h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: bf2cvtl2 v0.8h, v0.16h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f1cvtl v0.8h, v0.8h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: f1cvtl v0.8h, v0.8h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f1cvtl2 v0.8h, v0.16h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: f1cvtl2 v0.8h, v0.16h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f2cvtl v0.8h, v0.8h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: f2cvtl v0.8h, v0.8h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f2cvtl2 v0.8h, v0.16h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: f2cvtl2 v0.8h, v0.16h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtn  v31.8h, v31.4h, v31.4h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvtn  v31.8h, v31.4h, v31.4h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtn  v0.8s, v0.4s, v0.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fcvtn  v0.8s, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtn2  v0.16s, v0.4s, v0.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fcvtn2  v0.16s, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  v0.4h, v0.4s, v0.4s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  v0.4h, v0.4s, v0.4s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  v0.8h, v0.8s, v0.8s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fscale  v0.8h, v0.8s, v0.8s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  v0.2s, v0.2h, v0.2h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  v0.2s, v0.2h, v0.2h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  v0.4s, v31.4h, v0.4h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  v0.4s, v31.4h, v0.4h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  v0.2d, v31.2h, v0.2h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  v0.2d, v31.2h, v0.2h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
