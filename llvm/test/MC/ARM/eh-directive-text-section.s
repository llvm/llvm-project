@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -s -sd -sr | FileCheck %s

@ Check the .fnstart directive and the .fnend directive.

@ The .fnstart directive and .fnend directive should create an entry in
@ exception handling table.  For example, if the function is defined in .text
@ section, then there should be an entry in .ARM.exidx section.

	.syntax	unified

	.text
	.globl	func1
	.align	2
	.type	func1,%function
	.fnstart
func1:
	bx	lr
	.fnend



@-------------------------------------------------------------------------------
@ Check the .text section.
@-------------------------------------------------------------------------------
@ CHECK: Sections [
@ CHECK:   Section {

@-------------------------------------------------------------------------------
@ Check the index of .text section.  This will be used in .ARM.exidx.
@-------------------------------------------------------------------------------
@ CHECK:     Index: 2
@ CHECK-NEXT:     Name: .text
@ CHECK:     Type: SHT_PROGBITS (0x1)
@ CHECK:     Flags [ (0x6)
@ CHECK:       SHF_ALLOC (0x2)
@ CHECK:       SHF_EXECINSTR (0x4)
@ CHECK:     ]
@ CHECK:     SectionData (
@ CHECK:       0000: 1EFF2FE1                             |../.|
@ CHECK:     )
@ CHECK:   }


@-------------------------------------------------------------------------------
@ Check the name of the EXIDX section.  For the function in the .text section,
@ this should be .ARM.exidx.  It is incorrect to see .ARM.exidx.text here.
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .ARM.exidx
@ CHECK:     Type: SHT_ARM_EXIDX (0x70000001)
@ CHECK:     Flags [ (0x82)
@ CHECK:       SHF_ALLOC (0x2)
@ CHECK:       SHF_LINK_ORDER (0x80)
@ CHECK:     ]

@-------------------------------------------------------------------------------
@ Check the linked section of the EXIDX section.  This should be the index
@ of the .text section.
@-------------------------------------------------------------------------------
@ CHECK:     Link: 2

@-------------------------------------------------------------------------------
@ The first word should be the offset to .text.  The second word should be
@ 0xB0B0B080, which means compact model 0 is used (0x80) and the rest of the
@ word is filled with FINISH opcode (0xB0).
@-------------------------------------------------------------------------------
@ CHECK:     SectionData (
@ CHECK:       0000: 00000000 B0B0B080                    |........|
@ CHECK:     )
@ CHECK:   }
@ CHECK: ]

@-------------------------------------------------------------------------------
@ The first word should be relocated to the code address in .text section.
@ Besides, since this function is using compact model 0, thus we have to
@ add an relocation to __aeabi_unwind_cpp_pr0.
@-------------------------------------------------------------------------------
@ CHECK:     Relocations [
@ CHECK:       0x0 R_ARM_NONE __aeabi_unwind_cpp_pr0 0x0
@ CHECK:       0x0 R_ARM_PREL31 .text 0x0
@ CHECK:     ]
