// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of lower bound [-16, 15].

index z27.b, #-17, #-16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z27.b, #-17, #-16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

index z11.h, #-16, #-17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z11.h, #-16, #-17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

index z2.s, #16, #-16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z2.s, #16, #-16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

index z2.d, #-16, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z2.d, #-16, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

index z4.b, #-17, w28
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z4.b, #-17, w28
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

index z9.h, #16, w23
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z9.h, #16, w23
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

index z3.s, w10, #-17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z3.s, w10, #-17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

index z17.d, x9, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z17.d, x9, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid register

index z17.s, x9, w7
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z17.s, x9, w7
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

index z17.d, w9, w7
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-16, 15].
// CHECK-NEXT: index z17.d, w9, w7
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z21.d, p0/z, z28.d
index   z21.d, x10, x21
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: index   z21.d, x10, x21
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z21, z28
index   z21.d, x10, x21
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: index   z21.d, x10, x21
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23.d, p0/z, z30.d
index   z23.d, x13, #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: index   z23.d, x13, #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23, z30
index   z23.d, x13, #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: index   z23.d, x13, #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23.d, p0/z, z30.d
index   z23.d, #13, x8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: index   z23.d, #13, x8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23, z30
index   z23.d, #13, x8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: index   z23.d, #13, x8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
