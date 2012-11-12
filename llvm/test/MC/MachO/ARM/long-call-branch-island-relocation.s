@ RUN: llvm-mc -n -triple armv7-apple-darwin10 %s -filetype=obj -o %t.o
@ RUN: macho-dump --dump-section-data < %t.o | FileCheck %s

@ rdar://12359919

	.syntax unified
	.text

	.globl	_bar
	.align	2
	.code	16
	.thumb_func	_bar
_bar:
	push	{r7, lr}
	mov	r7, sp
	bl	_foo
	pop	{r7, pc}


_junk:
@ Make the _foo symbol sufficiently far away to force the 'bl' relocation
@ above to be out of range. On Darwin, the assembler deals with this by
@ generating an external relocation so the linker can create a branch
@ island.

  .space 20000000

  .section	__TEXT,initcode,regular,pure_instructions

	.globl	_foo
	.align	2
	.code	16
_foo:
	push	{r7, lr}
	mov	r7, sp
	pop	{r7, pc}


@ CHECK:  ('_relocations', [
@ CHECK:    # Relocation 0
@ CHECK:    (('word-0', 0x4),
@ CHECK:     ('word-1', 0x6d000002)),
@ CHECK:  ])
