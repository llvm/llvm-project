// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid immediates

fdup z0.h, #-0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.h, #-0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.s, #-0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.s, #-0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.d, #-0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.d, #-0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.h, #-64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.h, #-64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.s, #-64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.s, #-64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.d, #-64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.d, #-64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.h, #0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.h, #0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.s, #0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.s, #0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.d, #0.05859375   // r = -4, n = 15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.d, #0.05859375   // r = -4, n = 15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.h, #64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.h, #64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.s, #64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.s, #64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fdup z0.d, #64.00000000   // r = 5, n = 32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or floating-point constant
// CHECK-NEXT: fdup z0.d, #64.00000000   // r = 5, n = 32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
