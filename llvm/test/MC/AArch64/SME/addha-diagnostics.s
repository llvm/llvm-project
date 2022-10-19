// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme,+sme-i16i64 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid tile

addha za4.s, p0/m, p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: addha za4.s, p0/m, p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

addha za8.d, p0/m, p0/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: addha za8.d, p0/m, p0/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

addha za0p.s, p0/m, p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: addha za0p.s, p0/m, p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

addha za0h.s, p0/m, p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-7].d
// CHECK-NEXT: addha za0h.s, p0/m, p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

addha za0v.s, p0/m, p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-7].d
// CHECK-NEXT: addha za0v.s, p0/m, p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate

addha za0.s, p8/m, p0/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: addha za0.s, p8/m, p0/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

addha za0.s, p0/m, p8/m, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: addha za0.s, p0/m, p8/m, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

addha za0.d, p8/m, p0/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: addha za0.d, p8/m, p0/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

addha za0.d, p0/m, p8/m, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: addha za0.d, p0/m, p8/m, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
