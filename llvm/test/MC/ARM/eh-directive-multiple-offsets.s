@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd - | FileCheck %s

@ Check for different combination of .setfp, .pad, .save and .vsave.

	.syntax	unified

@-------------------------------------------------------------------------------
@ TEST1: Check .pad before .setfp directive.
@-------------------------------------------------------------------------------
	.section	.TEST1
	.globl	func1
	.type	func1,%function
	.align	2
	.fnstart
func1:
	.pad	#12
	sub	sp, sp, #12
	.setfp	fp, sp, #8
	add	fp, sp, #8
	sub	sp, fp, #8
	add	sp, sp, #12
	bx	lr
	.personality	__gxx_personality_v0
	.handlerdata
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST1
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0009B00                    |........|
@ CHECK:   )
@ CHECK: }



@-------------------------------------------------------------------------------
@ TEST2: Check .pad after .setfp directive.
@-------------------------------------------------------------------------------
	.section	.TEST2
	.globl	func2
	.type	func2,%function
	.align	2
	.fnstart
func2:
	.setfp	fp, sp, #8
	add	fp, sp, #8
	.pad	#12
	sub	sp, sp, #12
	add	sp, sp, #12
	sub	sp, fp, #8
	bx	lr
	.personality	__gxx_personality_v0
	.handlerdata
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST2
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0419B00                    |.....A..|
@ CHECK:   )
@ CHECK: }



@-------------------------------------------------------------------------------
@ TEST3: Check .setfp, .pad, .setfp directive.
@-------------------------------------------------------------------------------
	.section	.TEST3
	.globl	func3
	.type	func3,%function
	.align	2
	.fnstart
func3:
	@ prologue:
	.setfp	fp, sp, #4
	add	fp, sp, #4
	.pad	#8
	sub	sp, sp, #8
	.setfp	fp, sp, #4
	add	fp, sp, #4

	@ epilogue:
	add	sp, fp, #4
	bx	lr
	.personality	__gxx_personality_v0
	.handlerdata
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST3
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0009B00                    |........|
@ CHECK:   )
@ CHECK: }



@-------------------------------------------------------------------------------
@ TEST4: Check ".setfp fp, sp" and ".setfp fp, fp" directive.
@-------------------------------------------------------------------------------
	.section	.TEST4
	.globl	func4
	.type	func4,%function
	.align	2
	.fnstart
func4:
	@ prologue:
	.setfp	fp, sp, #8
	add	fp, sp, #8
	.setfp	fp, fp, #8
	add	fp, fp, #8

	@ epilogue:
	sub	sp, fp, #16
	bx	lr
	.personality	__gxx_personality_v0
	.handlerdata
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST4
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0439B00                    |.....C..|
@ CHECK:   )
@ CHECK: }



@-------------------------------------------------------------------------------
@ TEST5: Check .setfp, .save, .setfp directive.
@-------------------------------------------------------------------------------
	.section	.TEST5
	.globl	func5
	.type	func5,%function
	.align	2
	.fnstart
func5:
	@ prologue:
	.setfp	fp, sp, #16
	add	fp, sp, #16
	.save	{r4, r5, r6, r7, r8}
	push	{r4, r5, r6, r7, r8}
	.pad	#8
	add	sp, sp, #8
	.pad	#8
	sub	sp, sp, #8
	.save	{r9, r10}
	push	{r9, r10}
	.setfp	fp, sp, #24
	add	fp, sp, #24

	@ epilogue:
	sub	sp, fp, #24
	pop	{r9, r10}
	add	sp, sp, #16
	pop	{r4, r5, r6, r7, r8}
	bx	lr
	.personality	__gxx_personality_v0
	.handlerdata
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST5
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 80459B01 B0A40360           |.....E.....`|
@ CHECK:   )
@ CHECK: }
