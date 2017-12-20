// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.3a,-fullfp16 < %s 2>%t | FileCheck %s --check-prefix=CHECK --check-prefix=NO-FP16
// RUN: FileCheck --check-prefix=STDERR --check-prefix=STDERR-NO-FP16 %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.3a,+fullfp16 < %s 2>%t | FileCheck %s --check-prefix=CHECK --check-prefix=FP16
// RUN: FileCheck --check-prefix=STDERR --check-prefix=STDERR-FP16 %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a,-v8.3a,+fullfp16 < %s 2>&1 | FileCheck %s --check-prefix=NO-V83A


// ==== FCMLA vector ====
// Types
  fcmla v0.4h, v1.4h, v2.4h, #0
// FP16: fcmla   v0.4h, v1.4h, v2.4h, #0 // encoding: [0x20,0xc4,0x42,0x2e]
// STDERR-NO-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: fullfp16
// NO-V83A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.8h, v1.8h, v2.8h, #0
// FP16: fcmla   v0.8h, v1.8h, v2.8h, #0 // encoding: [0x20,0xc4,0x42,0x6e]
// STDERR-NO-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: fullfp16
// NO-V83A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.2s, v1.2s, v2.2s, #0
// CHECK: fcmla   v0.2s, v1.2s, v2.2s, #0 // encoding: [0x20,0xc4,0x82,0x2e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.4s, v1.4s, v2.4s, #0
// CHECK: fcmla   v0.4s, v1.4s, v2.4s, #0 // encoding: [0x20,0xc4,0x82,0x6e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.2d, v1.2d, v2.2d, #0
// CHECK: fcmla   v0.2d, v1.2d, v2.2d, #0 // encoding: [0x20,0xc4,0xc2,0x6e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a

// Rotations
  fcmla v0.2s, v1.2s, v2.2s, #0
// CHECK: fcmla   v0.2s, v1.2s, v2.2s, #0 // encoding: [0x20,0xc4,0x82,0x2e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.2s, v1.2s, v2.2s, #90
// CHECK: fcmla   v0.2s, v1.2s, v2.2s, #90 // encoding: [0x20,0xcc,0x82,0x2e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.2s, v1.2s, v2.2s, #180
// CHECK: fcmla   v0.2s, v1.2s, v2.2s, #180 // encoding: [0x20,0xd4,0x82,0x2e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.2s, v1.2s, v2.2s, #270
// CHECK: fcmla   v0.2s, v1.2s, v2.2s, #270 // encoding: [0x20,0xdc,0x82,0x2e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a

// Invalid rotations
  fcmla v0.2s, v1.2s, v2.2s, #1
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270.
  fcmla v0.2s, v1.2s, v2.2s, #360
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270.
  fcmla v0.2s, v1.2s, v2.2s, #-90
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270.

// ==== FCADD vector ====
// Types
  fcadd v0.4h, v1.4h, v2.4h, #90
// FP16: fcadd   v0.4h, v1.4h, v2.4h, #90 // encoding: [0x20,0xe4,0x42,0x2e]
// STDERR-NO-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: fullfp16
// NO-V83A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcadd v0.8h, v1.8h, v2.8h, #90
// FP16: fcadd   v0.8h, v1.8h, v2.8h, #90 // encoding: [0x20,0xe4,0x42,0x6e]
// STDERR-NO-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: fullfp16
// NO-V83A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcadd v0.2s, v1.2s, v2.2s, #90
// CHECK: fcadd   v0.2s, v1.2s, v2.2s, #90 // encoding: [0x20,0xe4,0x82,0x2e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcadd v0.4s, v1.4s, v2.4s, #90
// CHECK: fcadd   v0.4s, v1.4s, v2.4s, #90 // encoding: [0x20,0xe4,0x82,0x6e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcadd v0.2d, v1.2d, v2.2d, #90
// CHECK: fcadd   v0.2d, v1.2d, v2.2d, #90 // encoding: [0x20,0xe4,0xc2,0x6e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a

// Rotations
  fcadd v0.2s, v1.2s, v2.2s, #90
// CHECK: fcadd   v0.2s, v1.2s, v2.2s, #90 // encoding: [0x20,0xe4,0x82,0x2e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcadd v0.2s, v1.2s, v2.2s, #270
// CHECK: fcadd   v0.2s, v1.2s, v2.2s, #270 // encoding: [0x20,0xf4,0x82,0x2e]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a

// Invalid rotations
  fcadd v0.2s, v1.2s, v2.2s, #1
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270.
  fcadd v0.2s, v1.2s, v2.2s, #360
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270.
  fcadd v0.2s, v1.2s, v2.2s, #-90
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270.
  fcadd v0.2s, v1.2s, v2.2s, #0
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270.
  fcadd v0.2s, v1.2s, v2.2s, #180
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270.

// ==== FCMLA indexed ====
// Types
  fcmla v0.4h, v1.4h, v2.h[0], #0
// FP16: fcmla   v0.4h, v1.4h, v2.h[0], #0 // encoding: [0x20,0x10,0x42,0x2f]
// STDERR-NO-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: fullfp16
// NO-V83A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.8h, v1.8h, v2.h[0], #0
// FP16: fcmla   v0.8h, v1.8h, v2.h[0], #0 // encoding: [0x20,0x10,0x42,0x6f]
// STDERR-NO-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: fullfp16
// NO-V83A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.2s, v1.2s, v2.s[0], #0
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: invalid operand for instruction
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: invalid operand for instruction
  fcmla v0.4s, v1.4s, v2.s[0], #0
// CHECK: fcmla   v0.4s, v1.4s, v2.s[0], #0 // encoding: [0x20,0x10,0x82,0x6f]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.2d, v1.2d, v2.d[0], #0
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: invalid operand for instruction
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: invalid operand for instruction

// Rotations
  fcmla v0.4s, v1.4s, v2.s[0], #90
// CHECK: fcmla   v0.4s, v1.4s, v2.s[0], #90 // encoding: [0x20,0x30,0x82,0x6f]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.4s, v1.4s, v2.s[0], #180
// CHECK: fcmla   v0.4s, v1.4s, v2.s[0], #180 // encoding: [0x20,0x50,0x82,0x6f]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.4s, v1.4s, v2.s[0], #270
// CHECK: fcmla   v0.4s, v1.4s, v2.s[0], #270 // encoding: [0x20,0x70,0x82,0x6f]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a

// Valid indices
  fcmla v0.4h, v1.4h, v2.h[1], #0
// FP16: fcmla   v0.4h, v1.4h, v2.h[1], #0 // encoding: [0x20,0x10,0x62,0x2f]
// STDERR-NO-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: fullfp16
// NO-V83A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.8h, v1.8h, v2.h[3], #0
// FP16: fcmla   v0.8h, v1.8h, v2.h[3], #0 // encoding: [0x20,0x18,0x62,0x6f]
// STDERR-NO-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: fullfp16
// NO-V83A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
  fcmla v0.4s, v1.4s, v2.s[1], #0
// CHECK: fcmla   v0.4s, v1.4s, v2.s[1], #0 // encoding: [0x20,0x18,0x82,0x6f]
// NO-V83A: :[[@LINE-2]]:{{[0-9]*}}: error: instruction requires: armv8.3a

// Invalid indices
  fcmla v0.4h, v1.4h, v2.h[2], #0
// STDERR-NO-FP16: :[[@LINE-1]]:{{[0-9]*}}: error: invalid operand for instruction
// STDERR-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: vector lane must be an integer in range [0, 1].
  fcmla v0.8h, v1.8h, v2.h[4], #0
// STDERR-NO-FP16: :[[@LINE-1]]:{{[0-9]*}}: error: invalid operand for instruction
// STDERR-FP16: :[[@LINE-2]]:{{[0-9]*}}: error: vector lane must be an integer in range [0, 3].
  fcmla v0.4s, v1.4s, v2.s[2], #0
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: vector lane must be an integer in range [0, 1].

// Invalid rotations
  fcmla v0.4s, v1.4s, v2.s[0], #1
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270.
  fcmla v0.4s, v1.4s, v2.s[0], #360
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270.
  fcmla v0.4s, v1.4s, v2.s[0], #-90
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270.
