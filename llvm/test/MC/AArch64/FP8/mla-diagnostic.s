// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=fp8fma 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Element size extension incorrect
fmlalb  v0.8h, v0.8b, v0.8b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalb  v0.8h, v0.8b, v0.8b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalt  v31.8h, v31.16b, v31.16h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fmlalt  v31.8h, v31.16b, v31.16h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlallbb  v0.4s, v0, v31
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlallbb  v0.4s, v0, v31
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlallbt  v31.4b, v31.16b, v0.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlallbt  v31.4b, v31.16b, v0.16b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltb  v31.b, v31.16b, v0.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalltb  v31.b, v31.16b, v0.16b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltt  v0.4s, v0.16b, v31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalltt  v0.4s, v0.16b, v31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalt  v31.8s, v31.16b, v31.16b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fmlalt  v31.8s, v31.16b, v31.16b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


//--------------------------------------------------------------------------//
// Last Register range is between 0-7

fmlalltb  v0.4s, v31.16b, v8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalltb  v0.4s, v31.16b, v8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}$

fmlalt  v31.8h, v0.16b, v8.b[15]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmlalt  v31.8h, v0.16b, v8.b[15]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:$


// --------------------------------------------------------------------------//
// Out of range index
fmlalb  v31.8h, v0.16b, v0.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlalb  v31.8h, v0.16b, v0.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalt  v31.8h, v0.16b, v0.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlalt  v31.8h, v0.16b, v0.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlallbb  v31.4s, v0.16b, v7.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlallbb  v31.4s, v0.16b, v7.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltt  v31.4s, v0.16b, v7.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlalltt  v31.4s, v0.16b, v7.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmlalltb  v0.4s, v31.16b, v7.b[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlalltb  v0.4s, v31.16b, v7.b[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

fmlallbt  v0.4s, v31.16b, v0.b[16]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 15].
// CHECK-NEXT: fmlallbt  v0.4s, v31.16b, v0.b[16]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
