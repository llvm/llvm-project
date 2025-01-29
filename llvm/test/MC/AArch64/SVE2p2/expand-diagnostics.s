// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid element widths.

expand  z23.b, p3, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: expand  z23.b, p3, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.h, p3, z13.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: expand  z23.h, p3, z13.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.s, p3, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: expand  z23.s, p3, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.d, p3, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: expand  z23.d, p3, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.q, p3, z13.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: expand  z23.q, p3, z13.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid predicate operation

expand  z23.b, p3/z, z13.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: expand  z23.b, p3/z, z13.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.b, p3.b, z13.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.b, p3.b, z13.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.h, p3/m, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: expand  z23.h, p3/m, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.h, p3.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.h, p3.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.s, p3/z, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: expand  z23.s, p3/z, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.s, p3.s, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.s, p3.s, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.d, p3/m, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: expand  z23.d, p3/m, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.d, p3.d, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.d, p3.d, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Predicate not in restricted predicate range

expand  z23.b, p8, z13.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.b, p8, z13.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.b, p3.b, z13.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.b, p3.b, z13.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.h, p8, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.h, p8, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.h, p3.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.h, p3.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}

expand  z23.s, p8, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.s, p8, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

expand  z23.d, p8, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: expand  z23.d, p8, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31, z6
expand  z31.b, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: expand  z31.b, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.b, p0/z, z6.b
expand  z31.b, p0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: expand  z31.b, p0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
