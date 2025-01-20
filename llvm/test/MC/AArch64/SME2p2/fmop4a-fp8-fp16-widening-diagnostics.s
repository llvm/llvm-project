// RUN: not llvm-mc -triple=aarch64 -mattr=+sme2p2,+sme-f8f16 < %s 2>&1 | FileCheck %s

// Single vectors

fmop4a za0.d, z0.b, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za2.h, z0.b, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, z0.s, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.b..z14.b

fmop4a za0.h, z15.b, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.b..z14.b

fmop4a za0.h, z16.b, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.b..z14.b

fmop4a za0.h, z0.b, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.b..z30.b

fmop4a za0.h, z12.b, z17.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.b..z30.b

fmop4a za0.h, z12.b, z14.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.b..z30.b

fmop4a za0.h, z12.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.b..z30.b

// Single and multiple vectors

fmop4a za0.d, z0.b, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za2.h, z0.b, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, z0.s, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.b..z14.b

fmop4a za0.h, z1.b, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.b..z14.b

fmop4a za0.h, z16.b, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.b..z14.b

fmop4a za0.h, z0.b, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, z0.b, {z17.b-z18.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, z0.b, {z16.b-z18.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, z0.b, {z12.b-z13.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// Multiple and single vectors

fmop4a za0.d, {z0.b-z1.b}, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za2.h, {z0.b-z1.b}, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z0.s-z1.b}, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix

fmop4a za0.h, {z1.b-z2.b}, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za2.h, {z0.b-z2.b}, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z16.b-z17.b}, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, {z0.b-z1.b}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.b..z30.b

fmop4a za0.h, {z0.b-z1.b}, z17.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.b..z30.b

fmop4a za0.h, {z0.b-z1.b}, z12.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.b..z30.b

// Multiple vectors

fmop4a za0.d, {z0.b-z1.b}, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za2.h, {z0.b-z1.b}, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z0.s-z1.s}, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z1.b-z2.b}, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, {z0.b-z2.b}, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z18.b-z19.b}, {z16.b-z17.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, {z0.b-z1.b}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z0.b-z1.b}, {z19.b-z20.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, {z0.b-z1.b}, {z18.b-z20.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z0.b-z1.b}, {z10.b-z11.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types
