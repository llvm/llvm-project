// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate as counter register

pext    p0.h, pn3[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate-as-counter register expected pn8..pn15
// CHECK-NEXT: pext    p0.h, pn3[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid lane index

pext    p0.d, pn8[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pext    p0.d, pn8[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pext    p0.b, pn8[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pext    p0.b, pn8[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
