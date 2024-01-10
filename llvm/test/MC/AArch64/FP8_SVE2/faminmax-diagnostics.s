// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+faminmax 2>&1 < %s | FileCheck %s
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+faminmax 2>&1 < %s | FileCheck %s

// FAMIN:
// Invalid predicate register

famin z0.h, p8/m, z0.h, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: famin z0.h, p8/m, z0.h, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin z31.s, p7/z, z31.s, z30.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famin z31.s, p7/z, z31.s, z30.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

famin z23.h, p3/m, z23.h, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: famin z23.h, p3/m, z23.h, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin z23.b, p3/m, z23.d, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: famin z23.b, p3/m, z23.d, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Z register out of range

famin z31.s, p7/z, z31.s, z32.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famin z31.s, p7/z, z31.s, z32.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famin z0.d, p0/m, z0.d, z35.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famin z0.d, p0/m, z0.d, z35.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z20, z31
famin   z23.h, p3/m, z23.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx writing to a different destination
// CHECK-NEXT: famin   z23.h, p3/m, z23.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// FMAX:
// Invalid predicate register

famax z0.h, p8/m, z0.h, z1.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: famax z0.h, p8/m, z0.h, z1.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax z31.s, p7/z, z31.s, z30.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famax z31.s, p7/z, z31.s, z30.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector suffix

famax z23.h, p3/m, z23.h, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: famax z23.h, p3/m, z23.h, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax z23.b, p3/m, z23.d, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: famax z23.b, p3/m, z23.d, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Z register out of range

famax z31.s, p7/z, z31.s, z32.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famax z31.s, p7/z, z31.s, z32.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

famax z0.d, p0/m, z0.d, z35.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: famax z0.d, p0/m, z0.d, z35.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z20, z31
famax   z23.h, p3/m, z23.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx writing to a different destination
// CHECK-NEXT: famax   z23.h, p3/m, z23.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: