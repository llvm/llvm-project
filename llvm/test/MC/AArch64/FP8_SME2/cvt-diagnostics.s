// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+fp8 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Incorrect operand

f1cvt  { z0.h, z1.h }, z0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unknown token in expression
// CHECK-NEXT: f1cvt  { z0.h, z1.h }, z0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf1cvt  { z0, z1 }, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bf1cvt  { z0, z1 }, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf1cvtl { z0.b, z1.b }, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bf1cvtl { z0.b, z1.b }, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf2cvt  { z0.h, z1.h }, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bf2cvt  { z0.h, z1.h }, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bf2cvtl { z30.h}, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bf2cvtl { z30.h}, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f2cvt   { z0, z1.h }, {z0.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: f2cvt   { z0, z1.h }, {z0.b}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f2cvtl  z0.h, z1.h, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: f2cvtl  z0.h, z1.h, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvt    z31.b, { z30.h }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcvt    z31.b, { z30.h }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfcvt   z0.b, { z0.b, z1.b }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfcvt   z0.b, { z0.b, z1.b }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Incorrect range of vectors

bf1cvt { z1.h, z2.h }, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: bf1cvt { z1.h, z2.h }, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f1cvt  { z1.h, z0.h }, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: f1cvt  { z1.h, z0.h }, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

f1cvtl { z31.h, z0.h }, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: f1cvtl { z31.h, z0.h }, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvt   z31.b, { z29.s - z0.s }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: fcvt   z31.b, { z29.s - z0.s }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtn  z31.b, { z30.s - z1.s }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: fcvtn  z31.b, { z30.s - z1.s }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtn  z0.b, { z31.s - z2.s }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: fcvtn  z0.b, { z31.s - z2.s }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcvtn  z0.b, { z1.s - z4.s }
// CHECK: [[@LINE-1]]:{{[0-9]+}}:
// CHECK-NEXT: fcvtn  z0.b, { z1.s - z4.s }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
