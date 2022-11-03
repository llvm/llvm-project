// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector lists

pext {p0.h, p2.h}, pn8[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: pext {p0.h, p2.h}, pn8[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pext {p15.h, p1.h}, pn8[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: pext {p15.h, p1.h}, pn8[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pext {p0.h, p1.b}, pn8[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: pext {p0.h, p1.b}, pn8[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pext {p0.h, p1.h, p2.h, p3.h}, pn8[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: pext {p0.h, p1.h, p2.h, p3.h}, pn8[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pext {p0.h}, pn8[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: pext {p0.h}, pn8[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid predicate as counter register

pext    p0.h, pn3[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate-as-counter register expected pn8..pn15
// CHECK-NEXT: pext    p0.h, pn3[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid lane index

pext {p0.h, p1.h}, pn8[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: pext {p0.h, p1.h}, pn8[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pext {p0.h, p1.h}, pn8[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: pext {p0.h, p1.h}, pn8[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pext    p0.d, pn8[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pext    p0.d, pn8[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pext    p0.b, pn8[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: pext    p0.b, pn8[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
