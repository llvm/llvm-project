// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Invalid predicate suffix
fcpy z0.h, p0/z, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcpy z0.h, p0/z, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.s, p0/z, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcpy z0.s, p0/z, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.d, p0/z, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fcpy z0.d, p0/z, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediates

fcpy z0.h, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.h, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.s, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.s, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.d, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.d, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.h, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.h, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.s, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.s, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.d, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.d, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.h, p0/m, #0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.h, p0/m, #0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.s, p0/m, #0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.s, p0/m, #0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.d, p0/m, #0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.d, p0/m, #0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.h, p0/m, #64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.h, p0/m, #64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.s, p0/m, #64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.s, p0/m, #64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fcpy z0.d, p0/m, #64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fcpy z0.d, p0/m, #64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
