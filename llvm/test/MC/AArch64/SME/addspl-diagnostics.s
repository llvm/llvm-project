// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme  2>&1 < %s| FileCheck %s

// Immediate out of upper bound [-32, 31].
addspl x19, x14, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: addspl x19, x14, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// addspl requires an immediate, not a register.
addspl x19, x14, x15
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: addspl x19, x14, x15
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
