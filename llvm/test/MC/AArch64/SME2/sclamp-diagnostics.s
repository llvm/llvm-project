// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

sclamp {z0.b-z2.b}, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sclamp  {z0.b-z2.b}, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sclamp {z1.s-z2.s}, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element type
// CHECK-NEXT: sclamp {z1.s-z2.s}, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

sclamp {z0.h-z1.h}, z0.h, z4.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sclamp {z0.h-z1.h}, z0.h, z4.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
