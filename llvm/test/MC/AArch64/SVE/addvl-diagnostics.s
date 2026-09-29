// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Immediate out of upper bound [-32, 31].
addvl x3, x5, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: addvl x3, x5, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// addvl requires an immediate, not a register.
addvl x3, x5, x6
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: addvl x3, x5, x6
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
