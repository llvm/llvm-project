@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd --sr - | FileCheck %s

@ Check the compact pr1 model

	.syntax unified

	.section .TEST1
	.globl	func1
	.align	2
	.type	func1,%function
func1:
	.fnstart
	.save	{r4, r5, r11, lr}
	push	{r4, r5, r11, lr}
	add	r0, r1, r0
	.setfp	r11, sp, #8
	add	r11, sp, #8
	pop	{r4, r5, r11, pc}
	.fnend



@-------------------------------------------------------------------------------
@ Check .TEST1 section
@-------------------------------------------------------------------------------
@ CHECK: Sections [
@ CHECK:   Section {
@ CHECK:     Name: .TEST1
@ CHECK:     SectionData (
@ CHECK:       0000: 30482DE9 000081E0 08B08DE2 3088BDE8  |0H-.........0...|
@ CHECK:     )
@ CHECK:   }


@-------------------------------------------------------------------------------
@ Check .ARM.extab.TEST1 section
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .ARM.extab.TEST1
@-------------------------------------------------------------------------------
@ 0x81   = Compact model 1, personality routine: __aeabi_unwind_cpp_pr1
@ 0x9B   = $sp can be found in $r11
@ 0x41   = $sp = $sp - 8
@ 0x8483 = pop {r4, r5, r11, r14}
@ 0xB0   = finish
@-------------------------------------------------------------------------------
@ CHECK:     SectionData (
@ CHECK:       0000: 419B0181 B0B08384 00000000           |A...........|
@ CHECK:     )
@ CHECK:   }


@-------------------------------------------------------------------------------
@ Check .ARM.exidx.TEST1 section
@-------------------------------------------------------------------------------
@ CHECK:   Section {
@ CHECK:     Name: .ARM.exidx.TEST1
@ CHECK:     SectionData (
@ CHECK:       0000: 00000000 00000000                    |........|
@ CHECK:     )
@ CHECK:   }
@ CHECK: ]
@-------------------------------------------------------------------------------
@ The first word should be relocated to .TEST1 section, and the second word
@ should be relocated to .ARM.extab.TEST1 section.  Besides, there is
@ another relocation entry for __aeabi_unwind_cpp_pr1, so that the linker
@ will keep __aeabi_unwind_cpp_pr1.
@-------------------------------------------------------------------------------
@ CHECK:     Relocations [
@ CHECK:       0x0 R_ARM_NONE __aeabi_unwind_cpp_pr1 0x0
@ CHECK:       0x0 R_ARM_PREL31 .TEST1 0x0
@ CHECK:       0x4 R_ARM_PREL31 .ARM.extab.TEST1 0x0
@ CHECK:     ]
