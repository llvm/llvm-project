// RUN: not llvm-mc -triple=aarch64 -mattr=+sme2,+sve-bfscale 2>&1 < %s| FileCheck %s

// Multiple and single, 2 regs

bfmul   {z0.s-z1.s}, {z0.h-z1.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z1.h-z2.h}, {z0.h-z1.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

bfmul   {z0.h-z2.h}, {z0.h-z1.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z1.h}, {z0.s-z1.s}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z1.h}, {z1.h-z2.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

bfmul   {z0.h-z1.h}, {z0.h-z2.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z1.h}, {z0.h-z1.h}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

bfmul   {z0.h-z1.h}, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

// Multiple and single, 4 regs

bfmul   {z0.s-z3.s}, {z0.h-z3.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z1.h-z4.h}, {z0.h-z3.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types

bfmul   {z0.h-z4.h}, {z0.h-z3.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

bfmul   {z0.h-z3.h}, {z0.s-z3.s}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z3.h}, {z1.h-z4.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types

bfmul   {z0.h-z3.h}, {z0.h-z4.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

bfmul   {z0.h-z3.h}, {z0.h-z3.h}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

bfmul   {z0.h-z3.h}, {z0.h-z3.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

// Multiple, 2 regs

bfmul   {z0.s-z1.s}, {z0.h-z1.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z1.h-z2.h}, {z0.h-z1.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

bfmul   {z0.h-z2.h}, {z0.h-z1.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z1.h}, {z0.s-z1.s}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z1.h}, {z1.h-z2.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

bfmul   {z0.h-z1.h}, {z0.h-z2.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z1.h}, {z0.h-z1.h}, {z0.s-z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z1.h}, {z0.h-z1.h}, {z1.h-z2.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

bfmul   {z0.h-z1.h}, {z0.h-z1.h}, {z0.h-z2.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

// Multiple, 4 regs

bfmul   {z0.s-z3.s}, {z0.h-z3.h}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z1.h-z4.h}, {z0.h-z3.h}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types

bfmul   {z0.h-z4.h}, {z0.h-z3.h}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

bfmul   {z0.h-z3.h}, {z0.s-z3.s}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z3.h}, {z1.h-z4.h}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types

bfmul   {z0.h-z3.h}, {z0.h-z4.h}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

bfmul   {z0.h-z3.h}, {z0.h-z3.h}, {z0.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfmul   {z0.h-z3.h}, {z0.h-z3.h}, {z1.h-z4.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types

bfmul   {z0.h-z3.h}, {z0.h-z3.h}, {z0.h-z4.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
