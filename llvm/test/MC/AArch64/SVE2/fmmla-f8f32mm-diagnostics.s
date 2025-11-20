// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2,+f8f32mm   2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element width

fmmla   z21.b, z10.b, z21.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmmla   z21.b, z10.b, z21.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla   z21.h, z10.b, z21.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: f8f16mm
// CHECK-NEXT: fmmla   z21.h, z10.b, z21.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla   z21.d, z10.b, z21.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmmla   z21.d, z10.b, z21.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla   z21.s, z10.h, z21.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: sve-f16f32mm
// CHECK-NEXT: fmmla   z21.s, z10.h, z21.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla   z21.s, z10.s, z21.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: f32mm
// CHECK-NEXT: fmmla   z21.s, z10.s, z21.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}: