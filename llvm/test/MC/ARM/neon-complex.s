// RUN: not llvm-mc -triple thumb-none-linux-gnu -mattr=+v8.3a,+neon,+fullfp16 -show-encoding < %s 2>%t | FileCheck %s --check-prefix=THUMB --check-prefix=FP16-THUMB
// RUN: FileCheck --check-prefix=STDERR --check-prefix=NEON-STDERR <%t %s
// RUN: not llvm-mc -triple arm-none-linux-gnu -mattr=+v8.3a,+neon,+fullfp16 -show-encoding < %s 2>%t | FileCheck %s --check-prefix=ARM --check-prefix=FP16-ARM
// RUN: FileCheck --check-prefix=STDERR --check-prefix=NEON-STDERR <%t %s

// RUN: not llvm-mc -triple thumb-none-linux-gnu -mattr=+v8.3a,+neon,-fullfp16 -show-encoding < %s 2>%t | FileCheck %s --check-prefix=THUMB
// RUN: FileCheck --check-prefix=STDERR --check-prefix=NO-FP16-STDERR --check-prefix=NEON-STDERR <%t %s
// RUN: not llvm-mc -triple arm-none-linux-gnu -mattr=+v8.3a,+neon,-fullfp16 -show-encoding < %s 2>%t | FileCheck %s --check-prefix=ARM
// RUN: FileCheck --check-prefix=STDERR --check-prefix=NO-FP16-STDERR --check-prefix=NEON-STDERR <%t %s

// RUN: not llvm-mc -triple thumb-none-linux-gnu -mattr=+v8.3a,-neon,+fullfp16 -show-encoding < %s 2>%t
// RUN: FileCheck --check-prefix=STDERR --check-prefix=NO-NEON-STDERR <%t %s
// RUN: not llvm-mc -triple arm-none-linux-gnu -mattr=+v8.3a,-neon,+fullfp16 -show-encoding < %s 2>%t
// RUN: FileCheck --check-prefix=STDERR --check-prefix=NO-NEON-STDERR <%t %s

// RUN: not llvm-mc -triple thumb-none-linux-gnu -mattr=+v8.2a,+neon,+fullfp16 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=V82A
// RUN: not llvm-mc -triple arm-none-linux-gnu -mattr=+v8.2a,+neon,+fullfp16 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=V82A

/* ==== VCMLA vector ==== */

// Valid types
  vcmla.f16 d0, d1, d2, #0
// FP16-ARM: vcmla.f16       d0, d1, d2, #0  @ encoding: [0x02,0x08,0x21,0xfc]
// FP16-THUMB: vcmla.f16       d0, d1, d2, #0  @ encoding: [0x21,0xfc,0x02,0x08]
// NO-FP16-STDERR: :[[@LINE-3]]:{{[0-9]*}}: note: instruction requires: full half-float
// V82A: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-5]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f16 q0, q1, q2, #0
// FP16-ARM: vcmla.f16       q0, q1, q2, #0  @ encoding: [0x44,0x08,0x22,0xfc]
// FP16-THUMB: vcmla.f16       q0, q1, q2, #0  @ encoding: [0x22,0xfc,0x44,0x08]
// NO-FP16-STDERR: :[[@LINE-3]]:{{[0-9]*}}: note: instruction requires: full half-float
// V82A: :[[@LINE-4]]:{{[0-9]*}}: error: invalid instruction, any one of the following would fix this:
// V82A: :[[@LINE-5]]:{{[0-9]*}}: note: instruction requires: mve.fp
// V82A: :[[@LINE-6]]:{{[0-9]*}}: note: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-7]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f32 d0, d1, d2, #0
// ARM: vcmla.f32       d0, d1, d2, #0  @ encoding: [0x02,0x08,0x31,0xfc]
// THUMB: vcmla.f32       d0, d1, d2, #0  @ encoding: [0x31,0xfc,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f32 q0, q1, q2, #0
// ARM: vcmla.f32       q0, q1, q2, #0  @ encoding: [0x44,0x08,0x32,0xfc]
// THUMB: vcmla.f32       q0, q1, q2, #0  @ encoding: [0x32,0xfc,0x44,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: invalid instruction, any one of the following would fix this:
// V82A: :[[@LINE-4]]:{{[0-9]*}}: note: instruction requires: mve.fp
// V82A: :[[@LINE-5]]:{{[0-9]*}}: note: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-6]]:{{[0-9]*}}: error: instruction requires: NEON

// Valid rotations
  vcmla.f32 d0, d1, d2, #90
// ARM: vcmla.f32       d0, d1, d2, #90 @ encoding: [0x02,0x08,0xb1,0xfc]
// THUMB: vcmla.f32       d0, d1, d2, #90 @ encoding: [0xb1,0xfc,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f32 d0, d1, d2, #180
// ARM: vcmla.f32       d0, d1, d2, #180 @ encoding: [0x02,0x08,0x31,0xfd]
// THUMB: vcmla.f32       d0, d1, d2, #180 @ encoding: [0x31,0xfd,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f32 d0, d1, d2, #270
// ARM: vcmla.f32       d0, d1, d2, #270 @ encoding: [0x02,0x08,0xb1,0xfd]
// THUMB: vcmla.f32       d0, d1, d2, #270 @ encoding: [0xb1,0xfd,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON

// Invalid rotations
  vcmla.f32 d0, d1, d2, #-90
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction
  vcmla.f32 d0, d1, d2, #1
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction
  vcmla.f32 d0, d1, d2, #360
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction

/* ==== VCADD vector ==== */

// Valid types
  vcadd.f16 d0, d1, d2, #90
// FP16-ARM: vcadd.f16       d0, d1, d2, #90 @ encoding: [0x02,0x08,0x81,0xfc]
// FP16-THUMB: vcadd.f16       d0, d1, d2, #90 @ encoding: [0x81,0xfc,0x02,0x08]
// NO-FP16-STDERR: :[[@LINE-3]]:{{[0-9]*}}: note: instruction requires: full half-float
// V82A: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-5]]:{{[0-9]*}}: error: instruction requires: NEON
  vcadd.f16 q0, q1, q2, #90
// FP16-ARM: vcadd.f16       q0, q1, q2, #90 @ encoding: [0x44,0x08,0x82,0xfc]
// FP16-THUMB: vcadd.f16       q0, q1, q2, #90 @ encoding: [0x82,0xfc,0x44,0x08]
// NO-FP16-STDERR: :[[@LINE-3]]:{{[0-9]*}}: note: instruction requires: full half-float
// V82A: :[[@LINE-4]]:{{[0-9]*}}: error: invalid instruction, any one of the following would fix this:
// V82A: :[[@LINE-5]]:{{[0-9]*}}: note: instruction requires: mve.fp
// V82A: :[[@LINE-6]]:{{[0-9]*}}: note: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-7]]:{{[0-9]*}}: error: instruction requires: NEON
  vcadd.f32 d0, d1, d2, #90
// ARM: vcadd.f32       d0, d1, d2, #90 @ encoding: [0x02,0x08,0x91,0xfc]
// THUMB: vcadd.f32       d0, d1, d2, #90 @ encoding: [0x91,0xfc,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON
  vcadd.f32 q0, q1, q2, #90
// ARM: vcadd.f32       q0, q1, q2, #90 @ encoding: [0x44,0x08,0x92,0xfc]
// THUMB: vcadd.f32       q0, q1, q2, #90 @ encoding: [0x92,0xfc,0x44,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: invalid instruction, any one of the following would fix this:
// V82A: :[[@LINE-4]]:{{[0-9]*}}: note: instruction requires: mve.fp
// V82A: :[[@LINE-5]]:{{[0-9]*}}: note: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-6]]:{{[0-9]*}}: error: instruction requires: NEON

// Valid rotations
  vcadd.f32 d0, d1, d2, #270
// ARM: vcadd.f32       d0, d1, d2, #270 @ encoding: [0x02,0x08,0x91,0xfd]
// THUMB: vcadd.f32       d0, d1, d2, #270 @ encoding: [0x91,0xfd,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON

// Invalid rotations
  vcadd.f32 d0, d1, d2, #0
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction
  vcadd.f32 d0, d1, d2, #180
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction
  vcadd.f32 d0, d1, d2, #-90
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction
  vcadd.f32 d0, d1, d2, #1
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction
  vcadd.f32 d0, d1, d2, #360
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 90 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction


/* ==== VCMLA indexed ==== */

// Valid types
  vcmla.f16 d0, d1, d2[0], #0
// FP16-ARM: vcmla.f16       d0, d1, d2[0], #0 @ encoding: [0x02,0x08,0x01,0xfe]
// FP16-THUMB: vcmla.f16       d0, d1, d2[0], #0 @ encoding: [0x01,0xfe,0x02,0x08]
// NO-FP16-STDERR: :[[@LINE-3]]:{{[0-9]*}}: note: instruction requires: full half-float
// V82A: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-5]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f16 q0, q1, d2[0], #0
// FP16-ARM: vcmla.f16       q0, q1, d2[0], #0 @ encoding: [0x42,0x08,0x02,0xfe]
// FP16-THUMB: vcmla.f16       q0, q1, d2[0], #0 @ encoding: [0x02,0xfe,0x42,0x08]
// NO-FP16-STDERR: :[[@LINE-3]]:{{[0-9]*}}: note: instruction requires: full half-float
// V82A: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-5]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f32 d0, d1, d2[0], #0
// ARM: vcmla.f32       d0, d1, d2[0], #0 @ encoding: [0x02,0x08,0x81,0xfe]
// THUMB: vcmla.f32       d0, d1, d2[0], #0 @ encoding: [0x81,0xfe,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-5]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f32 q0, q1, d2[0], #0
// ARM: vcmla.f32       q0, q1, d2[0], #0 @ encoding: [0x42,0x08,0x82,0xfe]
// THUMB: vcmla.f32       q0, q1, d2[0], #0 @ encoding: [0x82,0xfe,0x42,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-5]]:{{[0-9]*}}: error: instruction requires: NEON

// Valid rotations
  vcmla.f32 d0, d1, d2[0], #90
// ARM: vcmla.f32       d0, d1, d2[0], #90 @ encoding: [0x02,0x08,0x91,0xfe]
// THUMB: vcmla.f32       d0, d1, d2[0], #90 @ encoding: [0x91,0xfe,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f32 d0, d1, d2[0], #180
// ARM: vcmla.f32       d0, d1, d2[0], #180 @ encoding: [0x02,0x08,0xa1,0xfe]
// THUMB: vcmla.f32       d0, d1, d2[0], #180 @ encoding: [0xa1,0xfe,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON
  vcmla.f32 d0, d1, d2[0], #270
// ARM: vcmla.f32       d0, d1, d2[0], #270 @ encoding: [0x02,0x08,0xb1,0xfe]
// THUMB: vcmla.f32       d0, d1, d2[0], #270 @ encoding: [0xb1,0xfe,0x02,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON

// Invalid rotations
  vcmla.f32 d0, d1, d2[0], #-90
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction
  vcmla.f32 d0, d1, d2[0], #1
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction
  vcmla.f32 d0, d1, d2[0], #360
// NEON-STDERR: :[[@LINE-1]]:{{[0-9]*}}: error: complex rotation must be 0, 90, 180 or 270
// NO-NEON-STDERR: :[[@LINE-2]]:{{[0-9]*}}: error: invalid instruction

// Valid indices
  vcmla.f16 d0, d1, d2[1], #0
// FP16-ARM: vcmla.f16       d0, d1, d2[1], #0 @ encoding: [0x22,0x08,0x01,0xfe]
// FP16-THUMB: vcmla.f16       d0, d1, d2[1], #0 @ encoding: [0x01,0xfe,0x22,0x08]
// V82A: :[[@LINE-3]]:{{[0-9]*}}: error: instruction requires: armv8.3a
// NO-NEON_STDERR: :[[@LINE-4]]:{{[0-9]*}}: error: instruction requires: NEON

// Invalid indices
// The text of these errors vary depending on whether fullfp16 is present.
  vcmla.f16 d0, d1, d2[2], #0
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error:
  vcmla.f32 d0, d1, d2[1], #0
// STDERR: :[[@LINE-1]]:{{[0-9]*}}: error:
