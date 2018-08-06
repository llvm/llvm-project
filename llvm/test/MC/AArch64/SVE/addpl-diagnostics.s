// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Immediate out of upper bound [-32, 31].
addpl x19, x14, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: addpl x19, x14, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// addpl requires an immediate, not a register.
addpl x19, x14, x15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: addpl x19, x14, x15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
