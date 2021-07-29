// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme,+sme-f64 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid tile
//
// expected: .s => za0-za3, .d => za0-za7

// non-widening

fmopa za4.s, p0/m, p0/m, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopa za4.s, p0/m, p0/m, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za8.d, p0/m, p0/m, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopa za8.d, p0/m, p0/m, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// widening

fmopa za4.s, p0/m, p0/m, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopa za4.s, p0/m, p0/m, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate (expected: p0-p7)

// non-widening

fmopa za0.s, p8/m, p0/m, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fmopa za0.s, p8/m, p0/m, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.s, p0/m, p8/m, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fmopa za0.s, p0/m, p8/m, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.d, p8/m, p0/m, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fmopa za0.d, p8/m, p0/m, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.d, p0/m, p8/m, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fmopa za0.d, p0/m, p8/m, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// widening

fmopa za0.s, p8/m, p0/m, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fmopa za0.s, p8/m, p0/m, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.s, p0/m, p8/m, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fmopa za0.s, p0/m, p8/m, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid predicate qualifier (expected: /m)

// non-widening

fmopa za0.s, p0/z, p0/m, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopa za0.s, p0/z, p0/m, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.s, p0/m, p0/z, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopa za0.s, p0/m, p0/z, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.d, p0/z, p0/m, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopa za0.d, p0/z, p0/m, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.d, p0/m, p0/z, z0.d, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopa za0.d, p0/m, p0/z, z0.d, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// widening

fmopa za0.s, p0/z, p0/m, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopa za0.s, p0/z, p0/m, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.s, p0/m, p0/z, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopa za0.s, p0/m, p0/z, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid ZPR type suffix
//
// expected: .s => .s (non-widening), .h (widening), .d => .d

// non-widening

fmopa za0.s, p0/m, p0/m, z0.b, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmopa za0.s, p0/m, p0/m, z0.b, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.s, p0/m, p0/m, z0.s, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmopa za0.s, p0/m, p0/m, z0.s, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.d, p0/m, p0/m, z0.b, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmopa za0.d, p0/m, p0/m, z0.b, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.d, p0/m, p0/m, z0.d, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmopa za0.d, p0/m, p0/m, z0.d, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// widening

fmopa za0.s, p0/m, p0/m, z0.b, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmopa za0.s, p0/m, p0/m, z0.b, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopa za0.s, p0/m, p0/m, z0.h, z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmopa za0.s, p0/m, p0/m, z0.h, z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
