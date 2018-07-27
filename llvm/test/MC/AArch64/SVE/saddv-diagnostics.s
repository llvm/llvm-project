// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid destination or source register.

saddv s0, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: saddv s0, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

saddv s0, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: saddv s0, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

saddv s0, p7, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: saddv s0, p7, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

saddv d0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: saddv d0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate

saddv d0, p8, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: saddv d0, p8, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: