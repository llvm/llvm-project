// RUN:     llvm-mc -triple   arm-none-none-eabi -show-encoding -mattr=+v8.3a,+fp-armv8 < %s 2>&1 | FileCheck %s --check-prefix=ARM
// RUN:     llvm-mc -triple thumb-none-none-eabi -show-encoding -mattr=+v8.3a,+fp-armv8 < %s 2>&1 | FileCheck %s --check-prefix=THUMB
// RUN: not llvm-mc -triple   arm-none-none-eabi -show-encoding -mattr=+v8.2a,+fp-armv8 < %s 2>&1 | FileCheck --check-prefix=REQ-V83 %s
// RUN: not llvm-mc -triple   arm-none-none-eabi -show-encoding -mattr=+v8.3a,-fp-armv8d16fp < %s 2>&1 | FileCheck --check-prefix=REQ-FP %s

  vjcvt.s32.f64 s1, d2
// ARM: vjcvt.s32.f64 s1, d2    @ encoding: [0xc2,0x0b,0xf9,0xee]
// THUMB: vjcvt.s32.f64 s1, d2    @ encoding: [0xf9,0xee,0xc2,0x0b]
// REQ-V83: error: instruction requires: armv8.3a
// REQ-FP: error: instruction requires: FPARMv8

  vjcvt.s32.f64 s17, d18
// ARM: vjcvt.s32.f64 s17, d18    @ encoding: [0xe2,0x8b,0xf9,0xee]
// THUMB: vjcvt.s32.f64 s17, d18    @ encoding: [0xf9,0xee,0xe2,0x8b]
// REQ-V83: error: instruction requires: armv8.3a
// REQ-FP: error: invalid instruction
