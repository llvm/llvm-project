// RUN: not llvm-mc -triple=aarch64 -mattr=+sme-mop4,+sme-f64f64 < %s 2>&1 | FileCheck %s

// FMOP4A

// Single vectors

fmop4a za0.s, z0.d, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4a za8.d, z0.d, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, z0.s, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4a za0.d, z15.d, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4a za0.d, z16.d, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4a za0.d, z0.d, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4a za0.d, z12.d, z17.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4a za0.d, z12.d, z14.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4a za0.d, z12.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

// Single and multiple vectors

fmop4a za0.s, z0.d, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4a za8.d, z0.d, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, z0.s, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4a za0.d, z1.d, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4a za0.d, z16.d, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4a za0.d, z0.d, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, z0.d, {z17.d-z18.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.d, z0.d, {z16.d-z18.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, z0.d, {z12.d-z13.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// Multiple and single vectors

fmop4a za0.s, {z0.d-z1.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za8.d, {z0.d-z1.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, {z0.s-z1.s}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, {z1.d-z2.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.d, {z0.d-z2.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, {z16.d-z17.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.d, {z0.d-z1.d}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4a za0.d, {z0.d-z1.d}, z17.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4a za0.d, {z0.d-z1.d}, z12.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

// Multiple vectors

fmop4a za0.s, {z0.d-z1.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za8.d, {z0.d-z1.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, {z0.s-z1.s}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, {z1.d-z2.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.d, {z0.d-z2.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, {z18.d-z19.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.d, {z0.d-z1.d}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, {z0.d-z1.d}, {z19.d-z20.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.d, {z0.d-z1.d}, {z16.d-z18.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.d, {z0.d-z1.d}, {z10.d-z11.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// FMOP4S

// Single vectors

fmop4s za0.s, z0.d, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4s za8.d, z0.d, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, z0.s, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4s za0.d, z15.d, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4s za0.d, z16.d, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4s za0.d, z0.d, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4s za0.d, z12.d, z17.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4s za0.d, z12.d, z14.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4s za0.d, z12.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

// Single and multiple vectors

fmop4s za0.s, z0.d, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4s za8.d, z0.d, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, z0.s, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4s za0.d, z1.d, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4s za0.d, z16.d, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.d..z14.d

fmop4s za0.d, z0.d, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, z0.d, {z17.d-z18.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.d, z0.d, {z16.d-z18.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, z0.d, {z12.d-z13.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// Multiple and single vectors

fmop4s za0.s, {z0.d-z1.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za8.d, {z0.d-z1.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, {z0.s-z1.s}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, {z1.d-z2.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.d, {z0.d-z2.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, {z16.d-z17.d}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.d, {z0.d-z1.d}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4s za0.d, {z0.d-z1.d}, z17.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

fmop4s za0.d, {z0.d-z1.d}, z12.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.d..z30.d

// Multiple vectors

fmop4s za0.s, {z0.d-z1.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za8.d, {z0.d-z1.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, {z0.s-z1.s}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, {z1.d-z2.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.d, {z0.d-z2.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, {z18.d-z19.d}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.d, {z0.d-z1.d}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, {z0.d-z1.d}, {z19.d-z20.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.d, {z0.d-z1.d}, {z16.d-z18.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.d, {z0.d-z1.d}, {z10.d-z11.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types
