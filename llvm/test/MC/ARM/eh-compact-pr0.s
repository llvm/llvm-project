@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -s -sd -sr > %t
@ RUN: FileCheck %s < %t
@ RUN: FileCheck --check-prefix=RELOC %s < %t

@ Check the compact pr0 model

	.syntax unified

	.section	.TEST1
	.globl	func1
	.align	2
	.type	func1,%function
func1:
	.fnstart
	.save	{r11, lr}
	push	{r11, lr}
	.setfp	r11, sp
	mov	r11, sp
	pop	{r11, lr}
	mov	pc, lr
	.fnend

	.section	.TEST2
	.globl	func2
	.align	2
	.type	func2,%function
func2:
	.fnstart
	.save	{r11, lr}
	push	{r11, lr}
	pop	{r11, pc}
	.fnend



@-------------------------------------------------------------------------------
@ Check .TEST1 section
@-------------------------------------------------------------------------------
@ CHECK: Sections [
@ CHECK:   Section {
@ CHECK:     Name: .TEST1
@ CHECK:     SectionData (
@ CHECK:       0000: 00482DE9 0DB0A0E1 0048BDE8 0EF0A0E1  |.H-......H......|
@ CHECK:     )
@ CHECK:   }


@-------------------------------------------------------------------------------
@ Check .ARM.exidx.TEST1 section
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .ARM.exidx.TEST1
@-------------------------------------------------------------------------------
@ 0x80   = Compact model 0, personality routine: __aeabi_unwind_cpp_pr0
@ 0x9B   = $sp can be found in $r11
@ 0x8480 = pop {r11, r14}
@-------------------------------------------------------------------------------
@ CHECK:     SectionData (
@ CHECK:       0000: 00000000 80849B80                    |........|
@ CHECK:     )
@ CHECK:   }
@-------------------------------------------------------------------------------
@ The first word should be relocated to .TEST1 section.  Besides, there is
@ another relocation entry for __aeabi_unwind_cpp_pr0, so that the linker
@ will keep __aeabi_unwind_cpp_pr0.
@-------------------------------------------------------------------------------
@ RELOC:   Section {
@ RELOC:     Name: .rel.ARM.exidx.TEST1
@ RELOC:     Relocations [
@ RELOC:       0x0 R_ARM_PREL31 .TEST1 0x0
@ RELOC:       0x0 R_ARM_NONE __aeabi_unwind_cpp_pr0 0x0
@ RELOC:     ]
@ RELOC:   }


@-------------------------------------------------------------------------------
@ Check .TEST2 section
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .TEST2
@ CHECK:     SectionData (
@ CHECK:       0000: 00482DE9 0088BDE8                    |.H-.....|
@ CHECK:     )
@ CHECK:   }
@-------------------------------------------------------------------------------
@ Check .ARM.exidx.TEST1 section
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .ARM.exidx.TEST2
@-------------------------------------------------------------------------------
@ 0x80   = Compact model 0, personality routine: __aeabi_unwind_cpp_pr0
@ 0x8480 = pop {r11, r14}
@ 0xB0   = finish
@-------------------------------------------------------------------------------
@ CHECK:     SectionData (
@ CHECK:       0000: 00000000 B0808480                    |........|
@ CHECK:     )
@ CHECK:   }
@-------------------------------------------------------------------------------
@ The first word should be relocated to .TEST2 section.  Besides, there is
@ another relocation entry for __aeabi_unwind_cpp_pr0, so that the linker
@ will keep __aeabi_unwind_cpp_pr0.
@-------------------------------------------------------------------------------
@ RELOC:   Section {
@ RELOC:     Name: .rel.ARM.exidx.TEST2
@ RELOC:     Relocations [
@ RELOC:       0x0 R_ARM_PREL31 .TEST2 0x0
@ RELOC:       0x0 R_ARM_NONE __aeabi_unwind_cpp_pr0 0x0
@ RELOC:     ]
@ RELOC:   }
