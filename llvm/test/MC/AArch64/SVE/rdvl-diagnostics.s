// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// Immediate out of upper bound [-32, 31].
rdvl x9, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: rdvl x9, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// rdvl requires an immediate, not a register.
rdvl x9, x10
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: rdvl x9, x10
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
