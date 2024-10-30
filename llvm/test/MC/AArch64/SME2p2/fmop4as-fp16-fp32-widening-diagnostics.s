// RUN: not llvm-mc -triple=aarch64 -mattr=+sme2p2 < %s 2>&1 | FileCheck %s

// FMOP4A

// Single vectors

fmop4a za0.d, z0.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za4.s, z0.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, z0.d, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4a za0.s, z15.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4a za0.s, z16.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4a za0.s, z0.h, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.s, z12.h, z17.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.s, z12.h, z14.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.s, z12.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

// Single and multiple vectors

fmop4a za0.d, z0.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za4.s, z0.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, z0.d, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4a za0.s, z1.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4a za0.s, z16.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4a za0.s, z0.h, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, z0.h, {z17.h-z18.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, z0.h, {z16.h-z18.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, z0.h, {z12.h-z13.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// Multiple and single vectors

fmop4a za0.d, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za4.s, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z0.d-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix

fmop4a za0.s, {z1.h-z2.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za4.s, {z0.h-z2.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z16.h-z17.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, {z0.h-z1.h}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.s, {z0.h-z1.h}, z17.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.s, {z0.h-z1.h}, z12.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

// Multiple vectors

fmop4a za0.d, {z0.h-z1.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za4.s, {z0.h-z1.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z0.d-z1.d}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z1.h-z2.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, {z0.h-z2.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z18.h-z19.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, {z0.h-z1.h}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z0.h-z1.h}, {z19.h-z20.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.s, {z0.h-z1.h}, {z18.h-z20.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.s, {z0.h-z1.h}, {z10.h-z11.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// FMOP4S

// Single vectors

fmop4a za0.d, z0.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za4.s, z0.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, z0.d, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4s za0.s, z15.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4s za0.s, z16.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4s za0.s, z0.h, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.s, z12.h, z17.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.s, z12.h, z14.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.s, z12.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

// Single and multiple vectors

fmop4s za0.d, z0.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za4.s, z0.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, z0.d, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4s za0.s, z1.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4s za0.s, z16.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register

fmop4s za0.s, z0.h, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, z0.h, {z17.h-z18.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, z0.h, {z16.h-z18.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, z0.h, {z12.h-z13.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// Multiple and single vectors

fmop4s za0.d, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za4.s, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z0.d-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix

fmop4s za0.s, {z1.h-z2.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z0.h-z2.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z16.h-z17.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z0.h-z1.h}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.s, {z0.h-z1.h}, z17.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.s, {z0.h-z1.h}, z12.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

// Multiple vectors

fmop4s za0.d, {z0.h-z1.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za4.s, {z0.h-z1.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z0.d-z1.d}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z1.h-z2.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z0.h-z2.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z18.h-z19.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z0.h-z1.h}, {z16.d-z17.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z0.h-z1.h}, {z19.h-z20.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.s, {z0.h-z1.h}, {z18.h-z20.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.s, {z0.h-z1.h}, {z10.h-z11.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types
