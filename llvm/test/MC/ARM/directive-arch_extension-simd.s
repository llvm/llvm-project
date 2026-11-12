@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7 -check-prefix CHECK
@ RUN: not llvm-mc -triple armv8-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK
@ RUN: not llvm-mc -triple thumbv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-V7 -check-prefix CHECK
@ RUN: not llvm-mc -triple thumbv8-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s -check-prefix CHECK

	.syntax unified

	.arch_extension simd
@ CHECK-V7: error: architectural extension 'simd' is not allowed for the current base architecture
@ CHECK-V7-NEXT: 	.arch_extension simd
@ CHECK-V7-NEXT:                     ^

	.type simd,%function
simd:
	vmaxnm.f32 s0, s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vminnm.f32 s0, s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8

	vmaxnm.f64 d0, d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vminnm.f64 d0, d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8

	vcvta.s32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvta.u32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvta.s32.f64 s0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvta.u32.f64 s0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtn.s32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtn.u32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtn.s32.f64 s0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtn.u32.f64 s0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtp.s32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtp.u32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtp.s32.f64 s0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtp.u32.f64 s0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtm.s32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtm.u32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtm.s32.f64 s0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vcvtm.u32.f64 s0, d0
@ CHECK-V7: error: instruction requires: FPARMv8

	vrintz.f32 s0, s1
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintz.f64 d0, d1
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintz.f32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintz.f64.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintr.f32 s0, s1
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintr.f64 d0, d1
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintr.f32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintr.f64.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintx.f32 s0, s1
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintx.f64 d0, d1
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintx.f32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintx.f64.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8

	vrinta.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrinta.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrinta.f32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrinta.f64.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintn.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintn.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintn.f32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintn.f64.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintp.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintp.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintp.f32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintp.f64.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintm.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintm.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintm.f32.f32 s0, s0
@ CHECK-V7: error: instruction requires: FPARMv8
	vrintm.f64.f64 d0, d0
@ CHECK-V7: error: instruction requires: FPARMv8

	.arch_extension nosimd
@ CHECK-V7: error: architectural extension 'simd' is not allowed for the current base architecture
@ CHECK-V7-NEXT: 	.arch_extension nosimd
@ CHECK-V7-NEXT:                     ^

	.type nosimd,%function
nosimd:
	vmaxnm.f32 s0, s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vminnm.f32 s0, s0, s0
@ CHECK: error: instruction requires: FPARMv8

	vmaxnm.f64 d0, d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vminnm.f64 d0, d0, d0
@ CHECK: error: instruction requires: FPARMv8

	vcvta.s32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vcvta.u32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vcvta.s32.f64 s0, d0
@ CHECK: error: instruction requires: FPARMv8
	vcvta.u32.f64 s0, d0
@ CHECK: error: instruction requires: FPARMv8
	vcvtn.s32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vcvtn.u32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vcvtn.s32.f64 s0, d0
@ CHECK: error: instruction requires: FPARMv8
	vcvtn.u32.f64 s0, d0
@ CHECK: error: instruction requires: FPARMv8
	vcvtp.s32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vcvtp.u32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vcvtp.s32.f64 s0, d0
@ CHECK: error: instruction requires: FPARMv8
	vcvtp.u32.f64 s0, d0
@ CHECK: error: instruction requires: FPARMv8
	vcvtm.s32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vcvtm.u32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vcvtm.s32.f64 s0, d0
@ CHECK: error: instruction requires: FPARMv8
	vcvtm.u32.f64 s0, d0
@ CHECK: error: instruction requires: FPARMv8

	vrintz.f32 s0, s1
@ CHECK: error: instruction requires: FPARMv8
	vrintz.f64 d0, d1
@ CHECK: error: instruction requires: FPARMv8
	vrintz.f32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrintz.f64.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vrintr.f32 s0, s1
@ CHECK: error: instruction requires: FPARMv8
	vrintr.f64 d0, d1
@ CHECK: error: instruction requires: FPARMv8
	vrintr.f32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrintr.f64.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vrintx.f32 s0, s1
@ CHECK: error: instruction requires: FPARMv8
	vrintx.f64 d0, d1
@ CHECK: error: instruction requires: FPARMv8
	vrintx.f32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrintx.f64.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8

	vrinta.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrinta.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vrinta.f32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrinta.f64.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vrintn.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrintn.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vrintn.f32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrintn.f64.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vrintp.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrintp.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vrintp.f32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrintp.f64.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vrintm.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrintm.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8
	vrintm.f32.f32 s0, s0
@ CHECK: error: instruction requires: FPARMv8
	vrintm.f64.f64 d0, d0
@ CHECK: error: instruction requires: FPARMv8

