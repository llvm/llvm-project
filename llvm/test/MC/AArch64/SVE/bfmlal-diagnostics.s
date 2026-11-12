// RUN: not llvm-mc -o - -triple=aarch64 -mattr=+sve,bf16  2>&1 %s | FileCheck %s

bfmlalb z0.S, z1.H, z7.H[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmlalb z0.S, z1.H, z7.H[8]
// CHECK-NEXT: ^

bfmlalb z0.S, z1.H, z8.H[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmlalb z0.S, z1.H, z8.H[7]
// CHECK-NEXT: ^

bfmlalt z0.S, z1.H, z7.H[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: bfmlalt z0.S, z1.H, z7.H[8]
// CHECK-NEXT: ^

bfmlalt z0.S, z1.H, z8.H[7]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfmlalt z0.S, z1.H, z8.H[7]
// CHECK-NEXT: ^

bfmlalt z0.S, z1.H, z7.2h[2]
// CHECK: error: invalid vector kind qualifier
// CHECK-NEXT: bfmlalt z0.S, z1.H, z7.2h[2]
// CHECK-NEXT:                     ^

bfmlalt z0.S, z1.H, z2.s[2]
// CHECK: error: Invalid restricted vector register, expected z0.h..z7.h
// CHECK-NEXT: bfmlalt z0.S, z1.H, z2.s[2]
// CHECK-NEXT:                     ^

bfmlalt z0.S, z1.s, z2.h[2]
// CHECK: error: invalid element width
// CHECK-NEXT: bfmlalt z0.S, z1.s, z2.h[2]
// CHECK-NEXT:               ^

movprfx z0.s, p0/m, z7.s
bfmlalt z0.s, z1.h, z2.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx
// CHECK-NEXT: bfmlalt z0.s, z1.h, z2.h
// CHECK-NEXT: ^
