// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p3 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element width

luti6 z0.h, zt0, z0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti6 z0.h, zt0, z0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 z0.s, zt0, z0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: luti6 z0.s, zt0, z0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 z0.d, zt0, z0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: luti6 z0.d, zt0, z0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/m, z7.h
luti6 z0.b, zt0, z1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 z0.b, zt0, z1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
luti6 z0.b, zt0, z1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 z0.b, zt0, z1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vectors/mis-matched registers/invalid index

luti6 { z0.h - z5.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: luti6 { z0.h - z5.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z0.b - z3.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: luti6 { z0.b - z3.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z0.h - z3.h }, { z0.h, z1.h }, { z0, z1 }[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti6 { z0.h - z3.h }, { z0.h, z1.h }, { z0, z1 }[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/m, z7.h
luti6 { z0.h - z3.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 { z0.h - z3.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
luti6 { z0.h - z3.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 { z0.h - z3.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Wrong striding/registers/index

luti6 { z0.h, z4.h, z8.h, z13.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must have the same sequential stride
// CHECK-NEXT: luti6 { z0.h, z4.h, z8.h, z13.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z1.h, z2.h, z3.h, z4.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: luti6 { z1.h, z2.h, z3.h, z4.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z0.b, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: luti6 { z0.b, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/m, z7.h
luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid registers

luti6 { z0.b - z5.b }, zt0, { z2 - z4 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: luti6 { z0.b - z5.b }, zt0, { z2 - z4 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z0.b - z3.b }, zt0, { z1 - z1 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: luti6 { z0.b - z3.b }, zt0, { z1 - z1 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z0.b - z5.b }, zt0, { z7 - z11 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid number of vectors
// CHECK-NEXT: luti6 { z0.b - z5.b }, zt0, { z7 - z11 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z0.b - z3.b }, zt1, { z1 - z3 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid lookup table, expected zt0
// CHECK-NEXT: luti6 { z0.b - z3.b }, zt1, { z1 - z3 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z4.b, z8.b, z12.b, z16.b}, zt0, { z2 - z5 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti6 { z4.b, z8.b, z12.b, z16.b}, zt0, { z2 - z5 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z17.b, z21.b, z25.b, z29.b}, zt0, { z2 - z5 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti6 { z17.b, z21.b, z25.b, z29.b}, zt0, { z2 - z5 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/m, z7.h
luti6 { z0.b - z3.b }, zt0, { z1 - z3 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 { z0.b - z3.b }, zt0, { z1 - z3 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
luti6 { z0.b - z3.b }, zt0, { z1 - z3 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 { z0.b - z3.b }, zt0, { z1 - z3 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Wrong striding/registers

luti6 { z1.b, z5.b, z9.b, z14.b }, zt0, { z0 - z2 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must have the same sequential stride
// CHECK-NEXT: luti6 { z1.b, z5.b, z9.b, z14.b }, zt0, { z0 - z2 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z1.b, z2.b, z3.b, z4.b }, zt0, { z0 - z2 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: luti6 { z1.b, z2.b, z3.b, z4.b }, zt0, { z0 - z2 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z20.b, z24.b, z28.b, z32.b }, zt0, { z0 - z2 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: luti6 { z20.b, z24.b, z28.b, z32.b }, zt0, { z0 - z2 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

luti6 { z1.h, z5.h, z9.h, z13.h }, zt0, { z0 - z2 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: luti6 { z1.h, z5.h, z9.h, z13.h }, zt0, { z0 - z2 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.h, p0/m, z7.h
luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z2 - z4 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z2 - z4 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z0 - z2 }
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z0 - z2 }
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
