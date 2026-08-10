@ RUN: llvm-mc -triple armv4t-eabi -mattr +d16 -filetype asm -o - %s 2>&1 | FileCheck %s

	.text
	.thumb

	.p2align 2

	.fpu vfpv3
	vldmia r0, {d16-d31}
@ CHECK: vldmia	r0, {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
@ CHECK-NOT: error: register expected

	.fpu vfpv4
	vldmia r0, {d16-d31}
@ CHECK: vldmia	r0, {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
@ CHECK-NOT: error: register expected

	.fpu neon
	vldmia r0, {d16-d31}
@ CHECK: vldmia	r0, {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
@ CHECK-NOT: error: register expected

	.fpu neon-vfpv4
	vldmia r0, {d16-d31}
@ CHECK: vldmia	r0, {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
@ CHECK-NOT: error: register expected

	.fpu neon-fp-armv8
	vldmia r0, {d16-d31}
@ CHECK: vldmia	r0, {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
@ CHECK-NOT: error: register expected

	.fpu crypto-neon-fp-armv8
	vldmia r0, {d16-d31}
@ CHECK: vldmia	r0, {d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
@ CHECK-NOT: error: register expected

