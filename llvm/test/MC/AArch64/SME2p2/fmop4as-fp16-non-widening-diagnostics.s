// RUN: not llvm-mc -triple=aarch64 -mattr=+sme-mop4,+sme-f16f16 < %s 2>&1 | FileCheck %s

// FMOP4A

// Single vectors

fmop4a za0.d, z0.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za2.h, z0.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, z0.s, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4a za0.h, z15.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4a za0.h, z16.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4a za0.h, z0.h, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.h, z12.h, z17.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.h, z12.h, z14.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.h, z12.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

// Single and multiple vectors

fmop4a za0.d, z0.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za2.h, z0.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, z0.s, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4a za0.h, z1.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4a za0.h, z16.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4a za0.h, z0.h, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, z0.h, {z17.h-z18.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, z0.h, {z12.h-z13.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// Multiple and single vectors

fmop4a za0.d, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za2.h, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z0.s-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix

fmop4a za0.h, {z1.h-z2.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, {z16.h-z17.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, {z0.h-z1.h}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.h, {z0.h-z1.h}, z17.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4a za0.h, {z0.h-z1.h}, z12.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

// Multiple vectors

fmop4a za0.d, {z0.h-z1.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4a za2.h, {z0.h-z1.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z0.s-z1.s}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z1.h-z2.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, {z18.h-z19.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, {z0.h-z1.h}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4a za0.h, {z0.h-z1.h}, {z19.h-z20.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4a za0.h, {z0.h-z1.h}, {z10.h-z11.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types


// FMOP4S

// Single vectors

fmop4s za0.d, z0.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za2.h, z0.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.h, z0.s, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4s za0.h, z15.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4s za0.h, z16.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4s za0.h, z0.h, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.h, z12.h, z17.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.h, z12.h, z14.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.h, z12.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

// Single and multiple vectors

fmop4s za0.d, z0.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za2.h, z0.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.h, z0.s, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4s za0.h, z1.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4s za0.h, z16.h, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h

fmop4s za0.h, z0.h, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.h, z0.h, {z17.h-z18.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.h, z0.h, {z12.h-z13.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

// Multiple and single vectors

fmop4s za0.d, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za2.h, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.h, {z0.s-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix

fmop4s za0.h, {z1.h-z2.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.h, {z16.h-z17.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.h, {z0.h-z1.h}, z16.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.h, {z0.h-z1.h}, z17.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

fmop4s za0.h, {z0.h-z1.h}, z12.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h

// Multiple vectors

fmop4s za0.d, {z0.h-z1.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand

fmop4s za2.h, {z0.h-z1.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.h, {z0.s-z1.s}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.h, {z1.h-z2.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.h, {z18.h-z19.h}, {z16.h-z17.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.h, {z0.h-z1.h}, {z16.s-z17.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmop4s za0.h, {z0.h-z1.h}, {z19.h-z20.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types

fmop4s za0.h, {z0.h-z1.h}, {z10.h-z11.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types
