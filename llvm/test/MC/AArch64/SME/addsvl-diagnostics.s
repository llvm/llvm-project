// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme  2>&1 < %s| FileCheck %s

// Immediate out of upper bound [-32, 31].
addsvl x3, x5, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: addsvl x3, x5, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// addsvl requires an immediate, not a register.
addsvl x3, x5, x6
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: addsvl x3, x5, x6
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
