@ RUN: llvm-mc < %s -triple armv7-unknown-unknown -show-encoding -mattr=+neon,+vfp4   | FileCheck %s --check-prefix=ARM
@ RUN: llvm-mc < %s -triple thumbv7-unknown-unknown -show-encoding -mattr=+neon,+vfp4 | FileCheck %s --check-prefix=THUMB
@ RUN: not llvm-mc < %s -triple thumbv7-unknown-unknown -show-encoding -mcpu=cortex-m4 > %t 2> %t2
@ RUN:     FileCheck %s < %t --check-prefix=THUMB_V7EM
@ RUN:     FileCheck %s < %t2 --check-prefix=THUMB_V7EM-ERRORS

@ ARM: vfma.f64 d16, d18, d17 @ encoding: [0xa1,0x0b,0xe2,0xee]
@ THUMB: vfma.f64 d16, d18, d17 @ encoding: [0xe2,0xee,0xa1,0x0b]
@ THUMB_V7EM-ERRORS: error: invalid instruction
@ THUMB_V7EM-ERRORS-NEXT: vfma.f64 d16, d18, d17
vfma.f64 d16, d18, d17

@ ARM: vfma.f32 s2, s4, s0 @ encoding: [0x00,0x1a,0xa2,0xee]
@ THUMB: vfma.f32 s2, s4, s0 @ encoding: [0xa2,0xee,0x00,0x1a]
@ THUMB_V7EM: vfma.f32 s2, s4, s0 @ encoding: [0xa2,0xee,0x00,0x1a]
vfma.f32 s2, s4, s0

@ ARM: vfma.f32 d16, d18, d17 @ encoding: [0xb1,0x0c,0x42,0xf2]
@ THUMB: vfma.f32 d16, d18, d17 @ encoding: [0x42,0xef,0xb1,0x0c]
@ THUMB_V7EM-ERRORS: error: invalid instruction
@ THUMB_V7EM-ERRORS-NEXT: vfma.f32 d16, d18, d17
vfma.f32 d16, d18, d17

@ ARM: vfma.f32 q2, q4, q0 @ encoding: [0x50,0x4c,0x08,0xf2]
@ THUMB: vfma.f32	q2, q4, q0 @ encoding: [0x08,0xef,0x50,0x4c]
@ THUMB_V7EM-ERRORS: error: instruction requires: mve.fp
@ THUMB_V7EM-ERRORS-NEXT: vfma.f32 q2, q4, q0
vfma.f32 q2, q4, q0

@ ARM: vfnma.f64 d16, d18, d17 @ encoding: [0xe1,0x0b,0xd2,0xee]
@ THUMB: vfnma.f64 d16, d18, d17 @ encoding: [0xd2,0xee,0xe1,0x0b]
@ THUMB_V7EM-ERRORS: error: invalid instruction
@ THUMB_V7EM-ERRORS-NEXT: vfnma.f64 d16, d18, d17
vfnma.f64 d16, d18, d17

@ ARM: vfnma.f32 s2, s4, s0 @ encoding: [0x40,0x1a,0x92,0xee]
@ THUMB: vfnma.f32 s2, s4, s0 @ encoding: [0x92,0xee,0x40,0x1a]
@ THUMB_V7EM: vfnma.f32 s2, s4, s0 @ encoding: [0x92,0xee,0x40,0x1a]
vfnma.f32 s2, s4, s0

@ ARM: vfms.f64 d16, d18, d17 @ encoding: [0xe1,0x0b,0xe2,0xee]
@ THUMB: vfms.f64 d16, d18, d17 @ encoding: [0xe2,0xee,0xe1,0x0b]
@ THUMB_V7EM-ERRORS: error: invalid instruction
@ THUMB_V7EM-ERRORS-NEXT: vfms.f64 d16, d18, d17
vfms.f64 d16, d18, d17

@ ARM: vfms.f32 s2, s4, s0 @ encoding: [0x40,0x1a,0xa2,0xee]
@ THUMB: vfms.f32 s2, s4, s0 @ encoding: [0xa2,0xee,0x40,0x1a]
@ THUMB_V7EM: vfms.f32 s2, s4, s0 @ encoding: [0xa2,0xee,0x40,0x1a]
vfms.f32 s2, s4, s0

@ ARM: vfms.f32 d16, d18, d17 @ encoding: [0xb1,0x0c,0x62,0xf2]
@ THUMB: vfms.f32 d16, d18, d17 @ encoding: [0x62,0xef,0xb1,0x0c]
@ THUMB_V7EM-ERRORS: error: invalid instruction
@ THUMB_V7EM-ERRORS-NEXT: vfms.f32 d16, d18, d17
vfms.f32 d16, d18, d17

@ ARM: vfms.f32 q2, q4, q0 @ encoding: [0x50,0x4c,0x28,0xf2]
@ THUMB: vfms.f32	q2, q4, q0 @ encoding: [0x28,0xef,0x50,0x4c]
@ THUMB_V7EM-ERRORS: error: instruction requires: mve.fp
@ THUMB_V7EM-ERRORS-NEXT: vfms.f32 q2, q4, q0
vfms.f32 q2, q4, q0

@ ARM: vfnms.f64 d16, d18, d17 @ encoding: [0xa1,0x0b,0xd2,0xee]
@ THUMB: vfnms.f64 d16, d18, d17 @ encoding: [0xd2,0xee,0xa1,0x0b]
@ THUMB_V7EM-ERRORS: error: invalid instruction
@ THUMB_V7EM-ERRORS-NEXT: vfnms.f64 d16, d18, d17
vfnms.f64 d16, d18, d17

@ ARM: vfnms.f32 s2, s4, s0 @ encoding: [0x00,0x1a,0x92,0xee]
@ THUMB: vfnms.f32 s2, s4, s0 @ encoding: [0x92,0xee,0x00,0x1a]
@ THUMB_V7EM: vfnms.f32 s2, s4, s0 @ encoding: [0x92,0xee,0x00,0x1a]
vfnms.f32 s2, s4, s0
