// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

umlsl za.s[w8, 0:1, vgx2], {z0.h-z2.h}, z0.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: umlsl za.s[w8, 0:1, vgx2], {z0.h-z2.h}, z0.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umlsl za.s[w9, 6:7], {z13.h-z16.h}, {z9.h-z12.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: umlsl za.s[w9, 6:7], {z13.h-z16.h}, {z9.h-z12.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid indexed-vector register

umlsl za.s[w11, 14:15], z31.h, z15.b[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: umlsl za.s[w11, 14:15], z31.h, z15.b[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umlsl za.s[w11, 6:7, vgx2], {z12.h-z13.h}, z31.h[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z15.h
// CHECK-NEXT: umlsl za.s[w11, 6:7, vgx2], {z12.h-z13.h}, z31.h[7]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select register

umlsl za.s[w7, 6:7], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: umlsl za.s[w7, 6:7], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umlsl za.s[w12, 6:7], {z12.h-z13.h}, z8.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be a register in range [w8, w11]
// CHECK-NEXT: umlsl za.s[w12, 6:7], {z12.h-z13.h}, z8.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector select offset

umlsl za.s[w11, 4:8], {z30.h-z31.h}, z15.h[15]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: umlsl za.s[w11, 4:8], {z30.h-z31.h}, z15.h[15]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umlsl za.s[w8, 10:12], z17.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: umlsl za.s[w8, 10:12], z17.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid Register Suffix

umlsl za.b[w8, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid matrix operand, expected suffix .s
// CHECK-NEXT: umlsl za.b[w8, 6:7, vgx2], {z12.h-z13.h}, {z8.h-z9.h}
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector lane index

umlsl za.s[w11, 6:7, vgx4], {z12.h-z15.h}, z8.h[64]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: umlsl za.s[w11, 6:7, vgx4], {z12.h-z15.h}, z8.h[64]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
