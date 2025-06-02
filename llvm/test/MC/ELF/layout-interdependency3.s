## This contrived .space example previously triggered "invalid number of bytes" error.
## https://github.com/llvm/llvm-project/issues/123402
# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: invalid number of bytes

	.section .p,"ax"
p_1st:
0:	pause
	lfence
	jmp	0b

	.section .q,"ax"
q_1st:
	addl	11,%eax
	addl	22,%eax

q_cli:
	cli
0:	pause
	lfence
	jmp	0b

	.section .p
	.space	(q_cli - q_1st) - (. - p_1st), 0xcc
	cli

	.section .q
q_sti:
	sti

	.section .p
	.space	(q_sti - q_1st) - (. - p_1st), 0xcc
	sti
	addl	33,%eax
	addl	44,%eax
p_nop:
	nop

	.section .q
0:	pause
	lfence
	jmp	0b
	.space	(p_nop - p_1st) - (. - q_1st), 0xcc
	nop
