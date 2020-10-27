@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd --sr --symbols - > %t
@ RUN: FileCheck %s < %t
@ RUN: FileCheck --check-prefix=RELOC %s < %t

@ Check whether the section is switched back properly.

@ The assembler should switch the section back to the corresponding section
@ after it have emitted the exception handling indices and tables.  In this
@ test case, we are checking whether the section is correct when .section
@ directives is used.

@ In this example, func1 and func2 should be defined in .TEST1 section.
@ It is incorrect if the func2 is in .text, .ARM.extab.TEST1, or
@ .ARM.exidx.TEST1 sections.

	.syntax	unified

	.section	.TEST1

	.globl	func1
	.align	2
	.type	func1,%function
	.fnstart
func1:
	bx	lr
	.personality	__gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
func2:
	bx	lr
	.personality	__gxx_personality_v0
	.handlerdata
	.fnend


@-------------------------------------------------------------------------------
@ Check the .text section.  This should be empty.
@-------------------------------------------------------------------------------
@ CHECK: Sections [
@ CHECK:   Section {
@ CHECK:     Name: .text
@ CHECK:     SectionData (
@ CHECK:     )
@ CHECK:   }


@-------------------------------------------------------------------------------
@ Check the .TEST1 section.  There should be two "bx lr" instructions.
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Index: 3
@ CHECK-NEXT:     Name: .TEST1
@ CHECK:     SectionData (
@ CHECK:       0000: 1EFF2FE1 1EFF2FE1                    |../.../.|
@ CHECK:     )
@ CHECK:   }



@-------------------------------------------------------------------------------
@ Check the .ARM.extab.TEST1 section.
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .ARM.extab.TEST1
@ CHECK:     SectionData (
@ CHECK:       0000: 00000000 B0B0B000 00000000 B0B0B000  |................|
@ CHECK:     )
@ CHECK:   }

@ RELOC:   Section {
@ RELOC:     Name: .rel.ARM.extab.TEST1
@ RELOC:     Relocations [
@ RELOC:       0x0 R_ARM_PREL31 __gxx_personality_v0 0x0
@ RELOC:       0x8 R_ARM_PREL31 __gxx_personality_v0 0x0
@ RELOC:     ]
@ RELOC:   }


@-------------------------------------------------------------------------------
@ Check the .ARM.exidx.TEST1 section.
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .ARM.exidx.TEST1
@ CHECK:     Link: 3
@-------------------------------------------------------------------------------
@ The first word should be the offset to .TEST1.
@ The second word should be the offset to .ARM.extab.TEST1
@-------------------------------------------------------------------------------
@ CHECK:     SectionData (
@ CHECK:       0000: 00000000 00000000 04000000 08000000  |................|
@ CHECK:     )
@ CHECK:   }
@-------------------------------------------------------------------------------
@ The first word of each entry should be relocated to .TEST1 section.
@ The second word of each entry should be relocated to
@ .ARM.extab.TESET1 section.
@-------------------------------------------------------------------------------

@ RELOC:   Section {
@ RELOC:     Name: .rel.ARM.exidx.TEST1
@ RELOC:     Relocations [
@ RELOC:       0x0 R_ARM_PREL31 .TEST1 0x0
@ RELOC:       0x4 R_ARM_PREL31 .ARM.extab.TEST1 0x0
@ RELOC:       0x8 R_ARM_PREL31 .TEST1 0x0
@ RELOC:       0xC R_ARM_PREL31 .ARM.extab.TEST1 0x0
@ RELOC:     ]
@ RELOC:   }


@-------------------------------------------------------------------------------
@ Check the symbols "func1" and "func2".  They should belong to .TEST1 section.
@-------------------------------------------------------------------------------
@ CHECK: Symbols [
@ CHECK:   Symbol {
@ CHECK:     Name: func1
@ CHECK:     Value: 0x0
@ CHECK:     Size: 0
@ CHECK:     Binding: Global (0x1)
@ CHECK:     Type: Function (0x2)
@ CHECK:     Other: 0
@ CHECK:     Section: .TEST1
@ CHECK:   }
@ CHECK:   Symbol {
@ CHECK:     Name: func2
@ CHECK:     Value: 0x4
@ CHECK:     Size: 0
@ CHECK:     Binding: Global (0x1)
@ CHECK:     Type: Function (0x2)
@ CHECK:     Other: 0
@ CHECK:     Section: .TEST1
@ CHECK:   }
@ CHECK: ]
