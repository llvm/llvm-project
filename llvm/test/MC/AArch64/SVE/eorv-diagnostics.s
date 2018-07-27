// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid destination or source register.

eorv d0, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: eorv d0, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eorv d0, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: eorv d0, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eorv d0, p7, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: eorv d0, p7, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eorv v0.2d, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: eorv v0.2d, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate

eorv h0, p8, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: eorv h0, p8, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: