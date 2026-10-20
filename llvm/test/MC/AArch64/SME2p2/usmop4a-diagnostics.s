// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-mop4,+sme-i16i64 2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid tile
//
// expected: .s => za0-za3, .d => za0-za7

usmop4a za4.s, z0.b, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: usmop4a za4.s, z0.b, z16.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

usmop4a za8.d, z0.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: usmop4a za8.d, z0.h, z16.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid first operand (expected z0..z15)

usmop4a za0.d, z16.h, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h
// CHECK-NEXT: usmop4a za0.d, z16.h, z16.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

usmop4a za0.d, {z16.h-z17.h}, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z0-z14, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: usmop4a za0.d, {z16.h-z17.h}, z16.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Invalid second operand (expected z16..z31)

usmop4a za0.d, z14.h, z14.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h
// CHECK-NEXT: usmop4a za0.d, z14.h, z14.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

usmop4a za0.d, z14.h, {z14.h-z15.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors in the range z16-z30, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: usmop4a za0.d, z14.h, {z14.h-z15.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid ZPR type suffix
//
// expected: .s => .b, .d => .h

usmop4a za3.s, z0.h, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.b..z14.b
// CHECK-NEXT: usmop4a za3.s, z0.h, z16.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

usmop4a za3.s, z0.b, z16.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.b..z30.b
// CHECK-NEXT: usmop4a za3.s, z0.b, z16.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

usmop4a za3.d, z0.h, z16.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z16.h..z30.h
// CHECK-NEXT: usmop4a za3.d, z0.h, z16.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

usmop4a za3.d, z0.s, z16.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected even register in z0.h..z14.h
// CHECK-NEXT: usmop4a za3.d, z0.s, z16.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
