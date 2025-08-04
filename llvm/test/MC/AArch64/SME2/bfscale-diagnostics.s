// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sve-bfscale 2>&1 < %s| FileCheck %s

// Multiple and single vector, 2 regs

bfscale  {z0.s-z1.s}, {z0.s-z1.s}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfscale  {z1.h-z2.h}, {z1.h-z2.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

bfscale  {z0.h-z2.h}, {z0.h-z2.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfscale  {z0.h-z1.h}, {z0.h-z1.h}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

bfscale  {z0.h-z1.h}, {z0.h-z1.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

bfscale  {z0.h-z1.h}, {z2.h-z3.h}, z8.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list

// Multiple and single vector, 4 regs

bfscale  {z0.s-z3.s}, {z0.s-z3.s}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfscale  {z1.h-z4.h}, {z1.h-z4.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types

bfscale  {z0.h-z4.h}, {z0.h-z4.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

bfscale  {z0.h-z3.h}, {z0.h-z3.h}, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

bfscale  {z0.h-z3.h}, {z0.h-z3.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h

bfscale  {z0.h-z3.h}, {z4.h-z7.h}, z8.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list

// Multiple vectors, 2 regs

bfscale  {z0.s-z1.s}, {z0.s-z1.s}, {z2.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfscale  {z1.h-z2.h}, {z1.h-z2.h}, {z2.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

bfscale  {z0.h-z2.h}, {z0.h-z4.h}, {z2.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

bfscale  {z0.h-z1.h}, {z0.h-z1.h}, {z2.s-z3.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfscale  {z0.h-z1.h}, {z0.h-z1.h}, {z28.h-z30.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfscale  {z0.h-z1.h}, {z0.h-z1.h}, {z29.h-z30.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types

bfscale  {z0.h-z1.h}, {z2.h-z3.h}, {z28.h-z29.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list

// Multiple vectors, 4 regs

bfscale  {z0.s-z3.s}, {z0.s-z3.s}, {z4.h-z7.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfscale  {z1.h-z4.h}, {z1.h-z4.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types

bfscale  {z0.h-z4.h}, {z0.h-z4.h}, {z4.h-z7.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error:  invalid number of vectors

bfscale  {z0.h-z3.h}, {z0.h-z3.h}, {z4.s-z7.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

bfscale  {z0.h-z3.h}, {z0.h-z3.h}, {z4.h-z8.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors

bfscale  {z0.h-z3.h}, {z0.h-z3.h}, {z5.h-z8.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types

bfscale  {z0.h-z3.h}, {z4.h-z7.h}, {z8.h-z11.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register list