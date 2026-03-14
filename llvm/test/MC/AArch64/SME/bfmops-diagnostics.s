// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid tile (expected: za0-za3)

bfmops za4.s, p0/m, p0/m, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmops za4.s, p0/m, p0/m, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate (expected: p0-p7)

bfmops za0.s, p8/m, p0/m, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: bfmops za0.s, p8/m, p0/m, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmops za0.s, p0/m, p8/m, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: bfmops za0.s, p0/m, p8/m, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate qualifier (expected: /m)

bfmops za0.s, p0/z, p0/m, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmops za0.s, p0/z, p0/m, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmops za0.s, p0/m, p0/z, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmops za0.s, p0/m, p0/z, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid ZPR type suffix (expected: .h)

bfmops za0.s, p0/m, p0/m, z0.b, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmops za0.s, p0/m, p0/m, z0.b, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfmops za0.s, p0/m, p0/m, z0.h, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfmops za0.s, p0/m, p0/m, z0.h, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
