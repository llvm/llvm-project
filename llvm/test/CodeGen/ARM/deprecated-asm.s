// REQUIRES: asserts
// RUN: llvm-mc < %s  -triple=armv4t-linux-gnueabi -filetype=obj -o %t.o -no-deprecated-warn -stats 2>&1 | FileCheck %s
// RUN: llvm-mc < %s  -triple=armv4t-linux-gnueabi -filetype=obj -o %t.o 2>&1 | FileCheck %s -check-prefix=WARN

	.text
	.syntax unified
	.eabi_attribute	67, "2.09"	@ Tag_conformance
	.cpu	arm7tdmi
	.eabi_attribute	6, 2	@ Tag_CPU_arch
	.eabi_attribute	8, 1	@ Tag_ARM_ISA_use
	.eabi_attribute	17, 1	@ Tag_ABI_PCS_GOT_use
	.eabi_attribute	20, 1	@ Tag_ABI_FP_denormal
	.eabi_attribute	21, 1	@ Tag_ABI_FP_exceptions
	.eabi_attribute	23, 3	@ Tag_ABI_FP_number_model
	.eabi_attribute	34, 0	@ Tag_CPU_unaligned_access
	.eabi_attribute	24, 1	@ Tag_ABI_align_needed
	.eabi_attribute	25, 1	@ Tag_ABI_align_preserved
	.eabi_attribute	38, 1	@ Tag_ABI_FP_16bit_format
	.eabi_attribute	18, 4	@ Tag_ABI_PCS_wchar_t
	.eabi_attribute	26, 2	@ Tag_ABI_enum_size
	.eabi_attribute	14, 0	@ Tag_ABI_PCS_R9_use
	.file	"t.c"
	.globl	foo
	.p2align	2
	.type	foo,%function
foo:                                    @ @foo
	.fnstart
@ BB#0:                                 @ %entry
	mov	r0, #0
	bx	lr
        stmia   r4!, {r12-r14}
.Lfunc_end0:
.Ltmp0:
	.size	foo, .Ltmp0-foo
	.cantunwind
	.fnend



// CHECK: Statistic
// CHECK-NOT: warning

// WARN: warning
