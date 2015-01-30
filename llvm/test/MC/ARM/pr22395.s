@ RUN: llvm-mc -triple armv4t-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

	.text
	.thumb

	.p2align 2

	.fpu neon
	vldmia r0, {d16-d31}

@ CHECK: vldmia	r0, {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
@ CHECK-NOT: error: instruction requires: VFP2

	.fpu vfpv3
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

	.fpu vfpv3-d16
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

	.fpu vfpv4
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

	.fpu vfpv4-d16
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

	.fpu fpv5-d16
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

	.fpu fp-armv8
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

	.fpu fp-armv8
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

	.fpu neon
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

	.fpu neon-vfpv4
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

	.fpu crypto-neon-fp-armv8
	vadd.f32 s1, s2, s3
@ CHECK: vadd.f32 s1, s2, s3
@ CHECK-NOT: error: instruction requires: VPF2

