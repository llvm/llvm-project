// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Register z32 does not exist.
sub z3.h, z26.h, z32.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sub z3.h, z26.h, z32.h

// Invalid element kind.
sub z4.h, z27.h, z31.x
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid sve vector kind qualifier
// CHECK-NEXT: sub z4.h, z27.h, z31.x

// Element size specifiers should match.
sub z0.h, z8.h, z8.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sub z0.h, z8.h, z8.b
