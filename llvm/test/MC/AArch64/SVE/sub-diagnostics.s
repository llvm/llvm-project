// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Register z32 does not exist.
sub z3.h, z26.h, z32.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sub z3.h, z26.h, z32.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid element kind.
sub z4.h, z27.h, z31.x
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid sve vector kind qualifier
// CHECK-NEXT: sub z4.h, z27.h, z31.x
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Element size specifiers should match.
sub z0.h, z8.h, z8.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sub z0.h, z8.h, z8.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Invalid predicate suffix '/a'
sub z29.d, p7/a, z29.d, z8.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expecting 'm' or 'z' predication
// CHECK-NEXT: sub z29.d, p7/a, z29.d, z8.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// Missing predicate suffix
sub z29.d, p7, z29.d, z8.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sub z29.d, p7, z29.d, z8.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

sub z26.b, p8/m, z26.b, z27.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: sub z26.b, p8/m, z26.b, z27.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub z14.h, p8/m, z14.h, z18.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: sub z14.h, p8/m, z14.h, z18.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub z30.s, p8/m, z30.s, z23.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: sub z30.s, p8/m, z30.s, z23.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sub z29.d, p8/m, z29.d, z3.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: sub z29.d, p8/m, z29.d, z3.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
