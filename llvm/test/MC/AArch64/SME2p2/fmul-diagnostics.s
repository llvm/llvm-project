
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 2>&1 < %s| FileCheck %s

// Multiple and single, 2 regs

fmul    {z0.b-z1.b}, {z0.h-z1.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z1.s-z2.s}, {z0.s-z1.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

fmul    {z0.d-z2.d}, {z0.d-z1.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.h-z1.h}, {z0.b-z1.b}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.s-z1.s}, {z1.s-z2.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

fmul    {z0.d-z1.d}, {z0.d-z2.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.h-z1.h}, {z0.h-z1.h}, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

fmul    {z0.s-z1.s}, {z0.s-z1.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.s..z15.s

// Multiple and single, 4 regs

fmul    {z0.b-z3.b}, {z0.h-z3.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z1.s-z3.s}, {z0.h-z3.h}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.d-z4.d}, {z0.d-z3.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

fmul    {z0.h-z3.h}, {z0.b-z3.b}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.s-z3.s}, {z1.s-z3.s}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.d-z3.d}, {z0.d-z4.d}, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

fmul    {z0.h-z3.h}, {z0.h-z3.h}, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

fmul    {z0.s-z3.s}, {z0.s-z3.s}, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.s..z15.s

// Multiple, 2 regs

fmul    {z0.b-z1.b}, {z0.h-z1.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z1.s-z2.s}, {z0.s-z1.s}, {z0.s-z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

fmul    {z0.d-z2.d}, {z0.d-z1.d}, {z0.d-z1.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.h-z1.h}, {z0.b-z1.b}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.s-z1.s}, {z1.s-z2.s}, {z0.s-z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

fmul    {z0.d-z1.d}, {z0.d-z2.d}, {z0.d-z1.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.h-z1.h}, {z0.h-z1.h}, {z0.b-z1.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.s-z1.s}, {z0.s-z1.s}, {z1.s-z2.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

fmul    {z0.d-z1.d}, {z0.d-z1.d}, {z0.d-z2.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

// Multiple, 4 regs

fmul    {z0.b-z3.b}, {z0.h-z3.h}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z1.s-z3.s}, {z0.s-z3.s}, {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.d-z4.d}, {z0.d-z3.d}, {z0.d-z3.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

fmul    {z0.h-z3.h}, {z0.b-z3.b}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.s-z3.s}, {z1.s-z3.s}, {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.d-z3.d}, {z0.d-z4.d}, {z0.d-z3.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

fmul    {z0.h-z3.h}, {z0.h-z3.h}, {z0.b-z3.b}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.s-z3.s}, {z0.s-z3.s}, {z1.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

fmul    {z0.d-z3.d}, {z0.d-z3.d}, {z0.d-z4.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
