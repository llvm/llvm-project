@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd --sr - | FileCheck %s

@ Check the .cantunwind directive

@ When a function contains a .cantunwind directive, we should create an entry
@ in corresponding .ARM.exidx, and its second word should be EXIDX_CANTUNWIND.

	.syntax	unified

	.text
	.globl	func1
	.align	2
	.type	func1,%function
	.fnstart
func1:
	bx	lr
	.cantunwind
	.fnend



@-------------------------------------------------------------------------------
@ Check .text section
@-------------------------------------------------------------------------------
@ CHECK: Sections [
@ CHECK:   Section {
@ CHECK:     Name: .text
@ CHECK:     SectionData (
@ CHECK:       0000: 1EFF2FE1                             |../.|
@ CHECK:     )
@ CHECK:   }


@-------------------------------------------------------------------------------
@ Check .ARM.exidx section
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .ARM.exidx
@-------------------------------------------------------------------------------
@ The first word should be the offset to .text.
@ The second word should be EXIDX_CANTUNWIND (01000000).
@-------------------------------------------------------------------------------
@ CHECK:     SectionData (
@ CHECK:       0000: 00000000 01000000                    |........|
@ CHECK:     )
@ CHECK:   }
@ CHECK: ]
@ CHECK:     Relocations [
@ CHECK:       0x0 R_ARM_PREL31 .text
@ CHECK:     ]
