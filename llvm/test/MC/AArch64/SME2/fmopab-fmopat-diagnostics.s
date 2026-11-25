// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f8f16 2>&1 < %s \
// RUN:   | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid ZA tile

fmopab za2.h, { z0.h, z1.h }, z0.b, z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopab za2.h, { z0.h, z1.h }, z0.b, z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopat za0.s, { z0.h, z1.h }, z0.b, z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected za[0-1].h
// CHECK-NEXT: fmopat za0.s, { z0.h, z1.h }, z0.b, z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

fmopab za0.h, { z1.h, z2.h }, z0.b, z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: fmopab za0.h, { z1.h, z2.h }, z0.b, z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopat za0.h, { z0.h, z2.h }, z0.b, z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmopat za0.h, { z0.h, z2.h }, z0.b, z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector element width

fmopab za0.h, { z0.h, z1.h }, z0.h, z20.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmopab za0.h, { z0.h, z1.h }, z0.h, z20.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Zk register

fmopat za0.h, { z0.h, z1.h }, z0.b, z19.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fmopat za0.h, { z0.h, z1.h }, z0.b, z19.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopab za0.h, { z0.h, z1.h }, z0.b, z19.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fmopab za0.h, { z0.h, z1.h }, z0.b, z19.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopat za0.h, { z0.h, z1.h }, z0.b, z24.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fmopat za0.h, { z0.h, z1.h }, z0.b, z24.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopab za0.h, { z0.h, z1.h }, z0.b, z24.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fmopab za0.h, { z0.h, z1.h }, z0.b, z24.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopat za0.h, { z0.h, z1.h }, z0.b, z27.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fmopat za0.h, { z0.h, z1.h }, z0.b, z27.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmopab za0.h, { z0.h, z1.h }, z0.b, z27.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted vector register, expected register in z20.b..z23.b or z28.b..z31.b
// CHECK-NEXT: fmopab za0.h, { z0.h, z1.h }, z0.b, z27.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
