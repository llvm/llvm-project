// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate not compatible with encode/decode function.

eon z5.b, z5.b, #0xfa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: eon z5.b, z5.b, #0xfa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eon z5.b, z5.b, #0xfff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: eon z5.b, z5.b, #0xfff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eon z5.h, z5.h, #0xfffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: eon z5.h, z5.h, #0xfffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eon z5.h, z5.h, #0xfffffff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: eon z5.h, z5.h, #0xfffffff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eon z5.s, z5.s, #0xfffffffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: eon z5.s, z5.s, #0xfffffffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eon z5.s, z5.s, #0xffffffffffffff9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: eon z5.s, z5.s, #0xffffffffffffff9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eon z15.d, z15.d, #0xfffffffffffffffa
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: expected compatible register or logical immediate
// CHECK-NEXT: eon z15.d, z15.d, #0xfffffffffffffffa
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Source and Destination Registers must match

eon z7.d, z8.d, #254
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: eon z7.d, z8.d, #254
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

eon z7.d, z8.d, #254
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: eon z7.d, z8.d, #254
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
eon     z0.d, z0.d, #0x6
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: eon     z0.d, z0.d, #0x6
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
