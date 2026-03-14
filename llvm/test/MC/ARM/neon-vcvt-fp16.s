@ RUN: llvm-mc -mcpu=cortex-r7 -triple arm -show-encoding < %s 2>&1| \
@ RUN:    FileCheck %s --check-prefix=CHECK-FP16
@ RUN: not llvm-mc -mcpu=cortex-r5 -triple arm -show-encoding < %s 2>&1 | \
@ RUN:    FileCheck %s --check-prefix=CHECK-NOFP16

@ CHECK-FP16: vcvtt.f32.f16	s7, s1         @ encoding: [0xe0,0x3a,0xf2,0xee]
@ CHECK-NOFP16: instruction requires: half-float conversions
	vcvtt.f32.f16	s7, s1
@ CHECK-FP16: vcvtt.f16.f32	s1, s7         @ encoding: [0xe3,0x0a,0xf3,0xee]
@ CHECK-NOFP16: instruction requires: half-float conversions
	vcvtt.f16.f32	s1, s7

@ CHECK-FP16: vcvtb.f32.f16	s7, s1         @ encoding: [0x60,0x3a,0xf2,0xee]
@ CHECK-NOFP16: instruction requires: half-float conversions
	vcvtb.f32.f16	s7, s1
@ CHECK-FP16: vcvtb.f16.f32	s1, s7         @ encoding: [0x63,0x0a,0xf3,0xee]
@ CHECK-NOFP16: instruction requires: half-float conversions
	vcvtb.f16.f32	s1, s7
