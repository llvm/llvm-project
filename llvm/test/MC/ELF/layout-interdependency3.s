## This contrived .space example previously triggered "invalid number of bytes" error.
## https://github.com/llvm/llvm-project/issues/123402
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# CHECK-LABEL: <p_1st>:
# CHECK:        e: cli
# CHECK-LABEL: <q_1st>:
# CHECK:        25: nop

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
