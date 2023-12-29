// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+fp8 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Incorrect operand

fscale  {z0.h-z1.h}, {z0.h-z1.h}, z0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: fscale  {z0.h-z1.h}, {z0.h-z1.h}, z0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  {z0.d-z1.d}, {z0.h-z1.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  {z0.d-z1.d}, {z0.h-z1.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  {z30.s-z31.s}, {z30.s-z31.s}, {z30 - z31}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  {z30.s-z31.s}, {z30.s-z31.s}, {z30 - z31}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  {z0.s-z3.s}, {z0.d-z3.d}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  {z0.s-z3.s}, {z0.d-z3.d}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  {z28.h-z31.h}, {z28-z31}, {z28.h-z31.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  {z28.h-z31.h}, {z28-z31}, {z28.h-z31.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  {z28.d-z31.d}, z28.d, {z28.d-z31.d}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  {z28.d-z31.d}, z28.d, {z28.d-z31.d}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  {z0.h-z1.h}, {z1.h-z4.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fscale  {z0.h-z1.h}, {z1.h-z4.h}, {z0.h-z1.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Incorrect range of vectors

fscale  {z0.h-z1.h}, {z1.h-z2.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: fscale  {z0.h-z1.h}, {z1.h-z2.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  {z0.h-z1.h}, {z31.h-z0.h}, {z0.h-z1.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: fscale  {z0.h-z1.h}, {z31.h-z0.h}, {z0.h-z1.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  {z2.h-z5.h}, {z0.h-z3.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: fscale  {z2.h-z5.h}, {z0.h-z3.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fscale  {z0.h-z3.h}, {z0.h-z3.h}, {z3.h-z6.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: fscale  {z0.h-z3.h}, {z0.h-z3.h}, {z3.h-z6.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
