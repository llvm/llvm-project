// RUN: not llvm-mc -triple=aarch64 -mattr=+sve,+sve-lrcpc < %s 2>&1 | FileCheck %s

ldapro z2, [x6, #177]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 32 in range [0, 8160].

stlro z2, [x6, #177]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 32 in range [0, 8160].

ldapro z2, [x6, #8161]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 32 in range [0, 8160].

stlro z2, [x6, #8161]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 32 in range [0, 8160].

ldapro z2, [x6, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 32 in range [0, 8160].

stlro z2, [x6, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be a multiple of 32 in range [0, 8160].
