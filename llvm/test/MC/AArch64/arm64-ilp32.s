// RUN: llvm-mc -target-abi=ilp32 -triple aarch64-non-linux-gnu -filetype=obj \
// RUN:  %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-ILP32 %s
// RUN: llvm-mc                   -triple aarch64-non-linux-gnu -filetype=obj \
// RUN:  %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-LP64 %s
	.text
	.file	"../projects/clang/test/Driver/arm64-ilp32.c"
	.globl	foo
	.align	2
	.type	foo,@function
foo:                                    // @foo
// %bb.0:                               // %entry
	sub	sp, sp, #16             // =16
// CHECK-ILP32: 0000000000000004 R_AARCH64_P32_ADR_PREL_PG_HI21 sizes
// CHECK-ILP32: 0000000000000008 R_AARCH64_P32_ADD_ABS_LO12_NC sizes
// CHECK-LP64:  0000000000000004 R_AARCH64_ADR_PREL_PG_HI21 sizes
// CHECK-LP64:  0000000000000008 R_AARCH64_ADD_ABS_LO12_NC sizes
	adrp	x8, sizes
	add	x8, x8, :lo12:sizes
	str	w0, [sp, #12]
	str	w1, [sp, #8]
	ldr		w0, [x8]
	add	sp, sp, #16             // =16
	ret
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo

	.type	sizes,@object           // @sizes
	.data
	.globl	sizes
	.align	2
sizes:
	.word	1                       // 0x1
	.word	2                       // 0x2
	.word	4                       // 0x4
	.word	4                       // 0x4
	.word	4                       // 0x4
	.size	sizes, 20
