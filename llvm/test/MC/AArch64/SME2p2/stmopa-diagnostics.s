// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid ZA register

stmopa  za4.s, {z30.b-z31.b}, z31.b, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stmopa  za4.s, {z30.b-z31.b}, z31.b, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za4.s, {z30.h-z31.h}, z31.h, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stmopa  za4.s, {z30.h-z31.h}, z31.h, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list operand

stmopa  za3.s, {z29.b-z30.b}, z31.b, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: stmopa  za3.s, {z29.b-z30.b}, z31.b, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.s, {z29.h-z30.h}, z31.h, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: stmopa  za3.s, {z29.h-z30.h}, z31.h, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid ZK register
stmopa  za3.s, {z28.b-z29.b}, z31.b, z19[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: stmopa  za3.s, {z28.b-z29.b}, z31.b, z19[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.s, {z28.b-z29.b}, z31.b, z24[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: stmopa  za3.s, {z28.b-z29.b}, z31.b, z24[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.s, {z28.b-z29.b}, z31.b, z27[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: stmopa  za3.s, {z28.b-z29.b}, z31.b, z27[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.s, {z28.h-z29.h}, z31.h, z19[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: stmopa  za3.s, {z28.h-z29.h}, z31.h, z19[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.s, {z28.h-z29.h}, z31.h, z24[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: stmopa  za3.s, {z28.h-z29.h}, z31.h, z24[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.s, {z28.h-z29.h}, z31.h, z27[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: stmopa  za3.s, {z28.h-z29.h}, z31.h, z27[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate

stmopa  za3.s, {z28.b-z29.b}, z31.b, z20[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: stmopa  za3.s, {z28.b-z29.b}, z31.b, z20[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.s, {z28.h-z29.h}, z31.h, z20[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: stmopa  za3.s, {z28.h-z29.h}, z31.h, z20[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid ZPR type suffix

stmopa  za0.h, {z28.b-z29.b}, z31.b, z20[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-3].s
// CHECK-NEXT: stmopa  za0.h, {z28.b-z29.b}, z31.b, z20[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za0.h, {z28.h-z29.h}, z31.h, z20[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-3].s
// CHECK-NEXT: stmopa  za0.h, {z28.h-z29.h}, z31.h, z20[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.s, {z28.s-z29.s}, z31.s, z20[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stmopa  za3.s, {z28.s-z29.s}, z31.s, z20[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.d, {z28.s-z29.s}, z31.s, z20[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-3].s
// CHECK-NEXT: stmopa  za3.d, {z28.s-z29.s}, z31.s, z20[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stmopa  za3.d, {z28.h-z29.h}, z31.h, z20[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-3].s
// CHECK-NEXT: stmopa  za3.d, {z28.h-z29.h}, z31.h, z20[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
