@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd --sr -r --symbols - | FileCheck %s

@ Check whether the section is switched back or not.

@ The assembler should emit the machine code of "func2" in .text section.
@ It is incorrect if the machine code is emitted in .ARM.exidx or .ARM.extab.
@ Besides, there should be two entries in .ARM.exidx section.

	.syntax	unified

	.text
	.globl	func1
	.align	2
	.type	func1,%function
	.fnstart
func1:
	bx	lr
	.fnend

	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
func2:
	bx	lr
	.fnend


@-------------------------------------------------------------------------------
@ Check the .text section.  There should be two "bx lr" instructions.
@-------------------------------------------------------------------------------
@ CHECK: Sections [
@ CHECK:   Section {
@ CHECK:     Name: .text
@ CHECK:     SectionData (
@ CHECK:       0000: 1EFF2FE1 1EFF2FE1                    |../.../.|
@ CHECK:     )
@ CHECK:   }


@-------------------------------------------------------------------------------
@ Check the .ARM.exidx section.
@ There should be two entries (two words per entry.)
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .ARM.exidx
@ CHECK:     SectionData (
@-------------------------------------------------------------------------------
@ The first word should be the offset to .text.  The second word should be
@ 0xB0B0B080, which means compact model 0 is used (0x80) and the rest of the
@ word is filled with FINISH opcode (0xB0).
@-------------------------------------------------------------------------------
@ CHECK:       0000: 00000000 B0B0B080 04000000 B0B0B080 |................|
@ CHECK:     )
@ CHECK:   }
@ CHECK: ]

@-------------------------------------------------------------------------------
@ The first word of each entry should be relocated to .text section.
@-------------------------------------------------------------------------------
@ CHECK:     Relocations [
@ CHECK:       0x0 R_ARM_PREL31 .text 0x0
@ CHECK:       0x0 R_ARM_NONE __aeabi_unwind_cpp_pr0 0x0
@ CHECK:       0x8 R_ARM_PREL31 .text 0x0
@ CHECK:     ]


@-------------------------------------------------------------------------------
@ Check the symbols "func1" and "func2".  They should belong to .text section.
@-------------------------------------------------------------------------------
@ CHECK: Symbols [
@ CHECK:   Symbol {
@ CHECK:     Name: func1
@ CHECK:     Section: .text
@ CHECK:   }
@ CHECK:   Symbol {
@ CHECK:     Name: func2
@ CHECK:     Section: .text
@ CHECK:   }
@ CHECK: ]
