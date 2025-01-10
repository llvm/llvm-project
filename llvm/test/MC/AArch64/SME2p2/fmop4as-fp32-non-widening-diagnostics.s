// RUN: not llvm-mc -triple=aarch64 -mattr=+sme2p2 < %s 2>&1 | FileCheck %s

// FMOP4A

// Single vectors

fmop4a za0.d, z0.s, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za4.s, z0.s, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, z0.d, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4a za0.s, z15.s, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4a za0.s, z16.s, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4a za0.s, z0.s, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4a za0.s, z12.s, z17.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4a za0.s, z12.s, z14.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4a za0.s, z12.s, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

// Single and multiple vectors

fmop4a za0.d, z0.s, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za4.s, z0.s, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, z0.d, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4a za0.s, z1.s, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4a za0.s, z16.s, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4a za0.s, z0.s, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, z0.s, {z17.s-z18.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, z0.s, {z16.s-z18.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, z0.s, {z12.s-z13.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// Multiple and single vectors

fmop4a za0.d, {z0.s-z1.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za4.s, {z0.s-z1.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z0.d-z1.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix

fmop4a za0.s, {z1.s-z2.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, {z2.s-z4.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z16.s-z17.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, {z0.s-z1.s}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4a za0.s, {z0.s-z1.s}, z17.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4a za0.s, {z0.s-z1.s}, z12.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

// Multiple vectors

fmop4a za0.d, {z0.s-z1.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za4.s, {z0.s-z1.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z0.d-z1.d}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z1.s-z2.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, {z2.s-z4.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z18.s-z19.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, {z0.s-z1.s}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z0.s-z1.s}, {z19.s-z20.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, {z0.s-z1.s}, {z16.s-z18.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z0.s-z1.s}, {z10.s-z11.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types


// FMOP4S

// Single vectors

fmop4s za0.d, z0.s, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za4.s, z0.s, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, z0.d, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4s za0.s, z15.s, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4s za0.s, z16.s, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4s za0.s, z0.s, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4s za0.s, z12.s, z17.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4s za0.s, z12.s, z14.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4s za0.s, z12.s, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

// Single and multiple vectors

fmop4s za0.d, z0.s, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za4.s, z0.s, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, z0.d, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4s za0.s, z1.s, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4s za0.s, z16.s, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.s..z14.s

fmop4s za0.s, z0.s, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, z0.s, {z17.s-z18.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, z0.s, {z16.s-z18.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, z0.s, {z12.s-z13.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// Multiple and single vectors

fmop4s za0.d, {z0.s-z1.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za4.s, {z0.s-z1.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z0.d-z1.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix

fmop4s za0.s, {z1.s-z2.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z2.s-z4.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z16.s-z17.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z0.s-z1.s}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4s za0.s, {z0.s-z1.s}, z17.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

fmop4s za0.s, {z0.s-z1.s}, z12.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.s..z30.s

// Multiple vectors

fmop4s za0.d, {z0.s-z1.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za4.s, {z0.s-z1.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z0.d-z1.d}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z1.s-z2.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z2.s-z4.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z18.s-z19.s}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z0.s-z1.s}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z0.s-z1.s}, {z19.s-z20.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z0.s-z1.s}, {z16.s-z18.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z0.s-z1.s}, {z10.s-z11.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

