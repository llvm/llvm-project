@ RUN: llvm-mc -n -triple armv7-apple-darwin10 %s -filetype=obj -o %t.o
@ RUN: llvm-readobj -r --expand-relocs < %t.o | FileCheck %s

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


@ CHECK: File: <stdin>
@ CHECK: Format: Mach-O arm
@ CHECK: Arch: arm
@ CHECK: AddressSize: 32bit
@ CHECK: Relocations [
@ CHECK:   Section __text {
@ CHECK:     Relocation {
@ CHECK:       Offset: 0x4
@ CHECK:       PCRel: 1
@ CHECK:       Length: 2
@ CHECK:       Type: ARM_THUMB_RELOC_BR22 (6)
@ CHECK:       Symbol: _foo (2)
@ CHECK:     }
@ CHECK:   }
@ CHECK: ]
