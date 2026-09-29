// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+faminmax 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Incorrect operand

famax   {z0.h-z1.h}, {z0.d-z1.d}, {z0.s-z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famax   {z0.h-z1.h}, {z0.d-z1.d}, {z0.s-z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax   {z28.s-z31.s}, {z28.h-z31.h}, {z28.d-z31.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famax   {z28.s-z31.s}, {z28.h-z31.h}, {z28.d-z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin   {z0.h-z1.h}, {z0.s-z1.s}, {z0.s-z1.s}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famin   {z0.h-z1.h}, {z0.s-z1.s}, {z0.s-z1.s}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Incorrect range of vectors

famax   {z1.d-z0.d}, {z0.d-z1.d}, {z0.d-z1.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: famax   {z1.d-z0.d}, {z0.d-z1.d}, {z0.d-z1.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax   {z0.h-z3.h}, {z1.h-z4.h}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: famax   {z0.h-z3.h}, {z1.h-z4.h}, {z0.h-z3.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax   {z0.h-z3.h}, {z0.h-z3.h}, {z2.h-z5.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: famax   {z0.h-z3.h}, {z0.h-z3.h}, {z2.h-z5.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax   {z3.h-z6.h}, {z0.h-z3.h}, {z0.h-z3.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: famax   {z3.h-z6.h}, {z0.h-z3.h}, {z0.h-z3.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin   {z30.h-z31.h}, {z31.h-z0.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: famin   {z30.h-z31.h}, {z31.h-z0.h}, {z0.h-z1.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin   {z29.d-z0.d}, {z0.d-z3.d}, {z0.d-z3.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: famin   {z29.d-z0.d}, {z0.d-z3.d}, {z0.d-z3.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin   {z0.d-z3.d}, {z30.d-z1.d}, {z0.d-z3.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: famin   {z0.d-z3.d}, {z30.d-z1.d}, {z0.d-z3.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin   {z0.d-z3.d}, {z0.d-z3.d}, {z28.d-z30.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famin   {z0.d-z3.d}, {z0.d-z3.d}, {z28.d-z30.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: