@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd --sr - > %t
@ RUN: FileCheck %s < %t
@ RUN: FileCheck --check-prefix=RELOC %s < %t

@ Check the .handlerdata directive (without .personality directive)

	.syntax unified

@-------------------------------------------------------------------------------
@ TEST1
@-------------------------------------------------------------------------------
	.section	.TEST1
	.globl	func1
	.align	2
	.type	func1,%function
	.fnstart
func1:
	bx	lr
	.handlerdata
	.fnend


@ CHECK:Section {
@ CHECK:  Name: .TEST1
@ CHECK:  SectionData (
@ CHECK:    0000: 1EFF2FE1                             |../.|
@ CHECK:  )
@ CHECK:}

@ CHECK:Section {
@ CHECK:  Name: .ARM.extab.TEST1
@ CHECK:  SectionData (
@ CHECK:    0000: B0B0B080                             |....|
@ CHECK:  )
@ CHECK:}

@ CHECK:Section {
@ CHECK:  Name: .ARM.exidx.TEST1
@ CHECK:  SectionData (
@ CHECK:    0000: 00000000 00000000                    |........|
@ CHECK:  )
@ CHECK:}
@-------------------------------------------------------------------------------
@ We should see a relocation entry to __aeabi_unwind_cpp_pr0, so that the
@ linker can keep __aeabi_unwind_cpp_pr0.
@-------------------------------------------------------------------------------
@ RELOC: Section {
@ RELOC:  Name: .rel.ARM.exidx.TEST1
@ RELOC:  Relocations [
@ RELOC:    0x0 R_ARM_NONE __aeabi_unwind_cpp_pr0 0x0
@ RELOC:    0x0 R_ARM_PREL31 .TEST1 0x0
@ RELOC:    0x4 R_ARM_PREL31 .ARM.extab.TEST1 0x0
@ RELOC:  ]
@ RELOC: }



@-------------------------------------------------------------------------------
@ TEST2
@-------------------------------------------------------------------------------
	.section	.TEST2
	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
func2:
@-------------------------------------------------------------------------------
@ Use a lot of unwind opcdes to get __aeabi_unwind_cpp_pr1.
@-------------------------------------------------------------------------------
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, r12}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, r12}
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, r12}
	.pad	#0x240
	sub	sp, sp, #0x240
	add	sp, sp, #0x240
	bx	lr
	.handlerdata
	.fnend



@ CHECK:Section {
@ CHECK:  Name: .TEST2
@ CHECK:  SectionData (
@ CHECK:    0000: F01F2DE9 F01FBDE8 09DD4DE2 09DD8DE2  |..-.......M.....|
@ CHECK:    0010: 1EFF2FE1                             |../.|
@ CHECK:  )
@ CHECK:}

@ CHECK:Section {
@ CHECK:  Name: .ARM.extab.TEST2
@ CHECK:  SectionData (
@ CHECK:    0000: 0FB20181 B0B0FF81                    |........|
@ CHECK:  )
@ CHECK:}

@ CHECK:Section {
@ CHECK:  Name: .ARM.exidx.TEST2
@ CHECK:  SectionData (
@ CHECK:    0000: 00000000 00000000                    |........|
@ CHECK:  )
@ CHECK:}
@-------------------------------------------------------------------------------
@ We should see a relocation entry to __aeabi_unwind_cpp_pr0, so that the
@ linker can keep __aeabi_unwind_cpp_pr0.
@-------------------------------------------------------------------------------
@ RELOC: Section {
@ RELOC:  Name: .rel.ARM.exidx.TEST2
@ RELOC:  Relocations [
@ RELOC:    0x0 R_ARM_NONE __aeabi_unwind_cpp_pr1 0x0
@ RELOC:    0x0 R_ARM_PREL31 .TEST2 0x0
@ RELOC:    0x4 R_ARM_PREL31 .ARM.extab.TEST2 0x0
@ RELOC:  ]
@ RELOC: }
