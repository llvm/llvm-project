// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+b16b16 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

bfminnm {z0.h-z1.h}, {z0.h-z2.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfminnm {z0.h-z1.h}, {z0.h-z2.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfminnm {z1.h-z2.h}, {z0.h-z1.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: bfminnm {z1.h-z2.h}, {z0.h-z1.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfminnm {z1.h-z4.h}, {z0.h-z3.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: bfminnm {z1.h-z4.h}, {z0.h-z3.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid single register

bfminnm {z0.h-z1.h}, {z2.h-z3.h}, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: bfminnm {z0.h-z1.h}, {z2.h-z3.h}, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

bfminnm {z0.h-z1.h}, {z2.h-z3.h}, z14.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: bfminnm {z0.h-z1.h}, {z2.h-z3.h}, z14.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfminnm {z0.h-z1.h}, {z2.s-z3.s}, z14.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfminnm {z0.h-z1.h}, {z2.s-z3.s}, z14.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfminnm {z0.h-z1.h}, {z2.h-z3.s}, z14.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: bfminnm {z0.h-z1.h}, {z2.h-z3.s}, z14.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
