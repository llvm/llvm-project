# RUN: llvm-mc --triple=x86_64-unknown-unknown -mcpu=skylake -filetype=obj < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

# The textual assembler *can't* default to autopadding as there's no syntax
# to selectively disable it just yet.
# CHECK: 0: pushq
# CHECK-NEXT: 1: movl
# CHECK-NEXT: 3: callq
# CHECK-NEXT: 8: callq
# CHECK-NEXT: d: callq
# CHECK-NEXT: 12: callq
# CHECK-NEXT: 17: callq
# No NOP
# CHECK-NEXT: 1c: testb
# CHECK-NEXT: 1f: je

	.text
	.globl	test
	.p2align	4, 0x90
	.type	test,@function
test:                                   # @test
# %bb.0:                                # %entry
	pushq	%rbx
	movl	%edi, %ebx
	callq	foo
	callq	foo
	callq	foo
	callq	foo
	callq	foo
	testb	$1, %bl
	je	.LBB0_2
# %bb.1:                                # %taken
	callq	foo
	popq	%rbx
	retq
.LBB0_2:                                # %untaken
	callq	bar
	popq	%rbx
	retq
