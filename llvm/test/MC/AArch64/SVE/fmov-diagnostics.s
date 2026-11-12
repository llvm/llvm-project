// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate suffix
fmov z0.h, p0/z, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmov z0.h, p0/z, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.s, p0/z, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmov z0.s, p0/z, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.d, p0/z, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmov z0.d, p0/z, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid immediates

fmov z0.h, #-0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.h, #-0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.s, #-0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.s, #-0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.d, #-0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.d, #-0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.h, #-64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.h, #-64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.s, #-64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.s, #-64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.d, #-64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.d, #-64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.h, #0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.h, #0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.s, #0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.s, #0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.d, #0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.d, #0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.h, #64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.h, #64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.s, #64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.s, #64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.d, #64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fmov z0.d, #64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.h, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.h, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.s, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.s, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.d, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.d, p0/m, #-0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.h, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.h, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.s, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.s, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.d, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.d, p0/m, #-64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.h, p0/m, #0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.h, p0/m, #0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.s, p0/m, #0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.s, p0/m, #0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.d, p0/m, #0.05859375    // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.d, p0/m, #0.05859375    // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.h, p0/m, #64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.h, p0/m, #64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.s, p0/m, #64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.s, p0/m, #64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmov z0.d, p0/m, #64.00000000    // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant 
// CHECK-NEXT: fmov z0.d, p0/m, #64.00000000    // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
fmov z0.d, #31.00000000
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fmov z0.d, #31.00000000
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
fmov z0.d, #31.00000000
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fmov z0.d, #31.00000000
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
