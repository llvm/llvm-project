// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-tmop,+sme-f8f32,+sme-f8f16,+sme-f16f16 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid ZA register (range)

ftmopa  za2.h, {z30.b-z31.b}, z31.b, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za2.h, {z30.b-z31.b}, z31.b, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za2.h, {z30.h-z31.h}, z31.h, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za2.h, {z30.h-z31.h}, z31.h, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za4.s, {z30.b-z31.b}, z31.b, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za4.s, {z30.b-z31.b}, z31.b, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za4.s, {z30.h-z31.h}, z31.h, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za4.s, {z30.h-z31.h}, z31.h, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za4.s, {z30.s-z31.s}, z31.s, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za4.s, {z30.s-z31.s}, z31.s, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid ZA register (type suffix)

ftmopa  za0.b, {z30.b-z31.b}, z31.b, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-3].s
// CHECK-NEXT: ftmopa  za0.b, {z30.b-z31.b}, z31.b, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.d, {z30.h-z31.h}, z31.h, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-3].s
// CHECK-NEXT: ftmopa  za0.d, {z30.h-z31.h}, z31.h, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.h, {z30.s-z31.s}, z31.s, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za0.h, {z30.s-z31.s}, z31.s, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list operand

ftmopa  za0.h, {z28.b-z31.b}, z31.b, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za0.h, {z28.b-z31.b}, z31.b, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.h, {z29.b-z30.b}, z31.b, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: ftmopa  za0.h, {z29.b-z30.b}, z31.b, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.h, {z28.h-z31.h}, z31.h, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za0.h, {z28.h-z31.h}, z31.h, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.h, {z29.h-z30.h}, z31.h, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: ftmopa  za0.h, {z29.h-z30.h}, z31.h, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.b-z31.b}, z31.b, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za3.s, {z28.b-z31.b}, z31.b, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z29.b-z30.b}, z31.b, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: ftmopa  za3.s, {z29.b-z30.b}, z31.b, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.h-z31.h}, z31.h, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za3.s, {z28.h-z31.h}, z31.h, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z29.h-z30.h}, z31.h, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: ftmopa  za3.s, {z29.h-z30.h}, z31.h, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.s-z31.s}, z31.s, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: ftmopa  za3.s, {z28.s-z31.s}, z31.s, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z29.s-z30.s}, z31.s, z31[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: ftmopa  za3.s, {z29.s-z30.s}, z31.s, z31[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid ZK register

ftmopa  za0.h, {z28.b-z29.b}, z31.b, z27[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za0.h, {z28.b-z29.b}, z31.b, z27[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.h, {z28.b-z29.b}, z31.b, z21.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za0.h, {z28.b-z29.b}, z31.b, z21.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.h, {z28.h-z29.h}, z31.h, z19[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za0.h, {z28.h-z29.h}, z31.h, z19[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.h, {z28.h-z29.h}, z31.h, z30.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za0.h, {z28.h-z29.h}, z31.h, z30.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.b-z29.b}, z31.b, z27[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za3.s, {z28.b-z29.b}, z31.b, z27[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.b-z29.b}, z31.b, z29.b[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za3.s, {z28.b-z29.b}, z31.b, z29.b[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.h-z29.h}, z31.h, z24[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za3.s, {z28.h-z29.h}, z31.h, z24[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.h-z29.h}, z31.h, z21.h[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za3.s, {z28.h-z29.h}, z31.h, z21.h[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.s-z29.s}, z31.s, z19[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za3.s, {z28.s-z29.s}, z31.s, z19[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.s-z29.s}, z31.s, z30.s[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20..z23 or z28..z31
// CHECK-NEXT: ftmopa  za3.s, {z28.s-z29.s}, z31.s, z30.s[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate

ftmopa  za0.h, {z28.b-z29.b}, z31.b, z20[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: ftmopa  za0.h, {z28.b-z29.b}, z31.b, z20[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.h, {z28.h-z29.h}, z31.h, z20[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: ftmopa  za0.h, {z28.h-z29.h}, z31.h, z20[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.b-z29.b}, z31.b, z20[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: ftmopa  za3.s, {z28.b-z29.b}, z31.b, z20[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.h-z29.h}, z31.h, z20[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: ftmopa  za3.s, {z28.h-z29.h}, z31.h, z20[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za3.s, {z28.s-z29.s}, z31.s, z20[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: ftmopa  za3.s, {z28.s-z29.s}, z31.s, z20[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid ZPR type suffix

ftmopa  za0.h, {z28.b-z29.b}, z31.h, z20[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ftmopa  za0.h, {z28.b-z29.b}, z31.h, z20[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.h, {z28.h-z29.h}, z31.s, z20[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ftmopa  za0.h, {z28.h-z29.h}, z31.s, z20[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.s, {z28.b-z29.b}, z31.h, z20[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ftmopa  za0.s, {z28.b-z29.b}, z31.h, z20[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.s, {z28.h-z29.h}, z31.s, z20[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ftmopa  za0.s, {z28.h-z29.h}, z31.s, z20[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ftmopa  za0.s, {z28.s-z29.s}, z31.h, z20[3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: ftmopa  za0.s, {z28.s-z29.s}, z31.h, z20[3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


