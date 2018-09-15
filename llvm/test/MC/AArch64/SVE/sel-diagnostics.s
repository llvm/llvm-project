// RUN: not llvm-mc -triple=aarch64-none-linux-gnu -show-encoding -mattr=+sve  2>&1 < %s | FileCheck %s


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z28.b, p7/z, z30.b
sel     z28.b, p7, z13.b, z8.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sel     z28.b, p7, z13.b, z8.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z23, z30
sel     z23.b, p11, z13.b, z8.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: sel     z23.b, p11, z13.b, z8.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
