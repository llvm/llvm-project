// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid predicate

rbit  z0.d, p8/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: rbit  z0.d, p8/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
