// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme  2>&1 < %s| FileCheck %s

// Immediate out of upper bound [-32, 31].
rdsvl x9, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: rdsvl x9, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// rdsvl requires an immediate, not a register.
rdsvl x9, x10
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: rdsvl x9, x10
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
