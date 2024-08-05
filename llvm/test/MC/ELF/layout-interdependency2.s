## Contrived .zero directive example, simplified from the Linux kernel use case,
## which requires multiple iterations to converge.
## https://github.com/llvm/llvm-project/issues/100283
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# CHECK:       80: jne 0x0 <.text>
# CHECK-EMPTY:

	.text
.Ltmp0:
.Ltmp1:
	cli
	popq	%rdi
text1:
	.zero	(.Ltmp2-.Ltmp3)
	.section	"","ax",@progbits
.Ltmp3:
	movq	$0, %rax
.Ltmp4:
.Ltmp5:
	.section	.discard.intra_function_calls,"ax",@progbits
	.long	.Ltmp5
	.section	"","ax",@progbits
	callq	.Ltmp6
	int3
.Ltmp7:
.Ltmp8:
	.section	.discard.intra_function_calls,"ax",@progbits
	.long	.Ltmp8
	.section	"","ax",@progbits
	callq	.Ltmp6
	int3
.Ltmp6:
	addq	$0, %rsp
	decq	%rax
	jne	.Ltmp4
	lfence
	movq	$-1, %gs:pcpu_hot+6

.Ltmp2:
	.text
text2:

	.zero	(.Ltmp9-.Ltmp10)
	.section	"","ax",@progbits
.Ltmp10:
	jmp	.Ltmp11
.Ltmp9:
	.text
text3:

.Ltmp12:
	.zero	(.Ltmp13-.Ltmp14)
	.section	"","ax",@progbits
.Ltmp14:
	callq	entry_untrain_ret
.Ltmp13:
	.text

	.zero	(.Ltmp15-.Ltmp16)
	.section	"","ax",@progbits
.Ltmp16:
	xorl	%eax, %eax
	btsq	$63, %rax
	movq	%rax, %gs:pcpu_hot+6

.Ltmp15:
	.text

	popq	%r12
	popq	rbp
	jmp	__x86_return_thunk
	movl	936(%rdi), %eax
	cmpl	%gs:x86_spec_ctrl_current, %eax
	je	.Ltmp0
	movl	edx, %edx
	wrmsr
	jmp	.Ltmp0
.Ltmp11:
	movl	$72, %ecx
	jmp	.Ltmp12
	cmpb	$0, kvm_rebooting
	jne	.Ltmp1
