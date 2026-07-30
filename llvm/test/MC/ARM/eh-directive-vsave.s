@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd --sr - | FileCheck %s

@ Check the .vsave directive

@ The .vsave directive records the VFP registers which are pushed to the
@ stack.  There are two different opcodes:
@
@     0xC800: pop d[(16+x+y):(16+x)]    @ d[16+x+y]-d[16+x] must be consecutive
@     0xC900: pop d[(x+y):x]            @ d[x+y]-d[x] must be consecutive


	.syntax unified

@-------------------------------------------------------------------------------
@ TEST1
@-------------------------------------------------------------------------------
	.section	.TEST1
	.globl	func1a
	.align	2
	.type	func1a,%function
	.fnstart
func1a:
	.vsave	{d0}
	vpush	{d0}
	vpop	{d0}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func1b
	.align	2
	.type	func1b,%function
	.fnstart
func1b:
	.vsave	{d0, d1, d2, d3}
	vpush	{d0, d1, d2, d3}
	vpop	{d0, d1, d2, d3}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func1c
	.align	2
	.type	func1c,%function
	.fnstart
func1c:
	.vsave	{d0, d1, d2, d3, d4, d5, d6, d7}
	vpush	{d0, d1, d2, d3, d4, d5, d6, d7}
	vpop	{d0, d1, d2, d3, d4, d5, d6, d7}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func1d
	.align	2
	.type	func1d,%function
	.fnstart
func1d:
	.vsave	{d2, d3, d4, d5, d6, d7}
	vpush	{d2, d3, d4, d5, d6, d7}
	vpop	{d2, d3, d4, d5, d6, d7}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST1
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B000C900 00000000 B003C900  |................|
@ CHECK:     0010: 00000000 B007C900 00000000 B025C900  |.............%..|
@ CHECK:   )
@ CHECK: }



@-------------------------------------------------------------------------------
@ TEST2
@-------------------------------------------------------------------------------
	.section	.TEST2
	.globl	func2a
	.align	2
	.type	func2a,%function
	.fnstart
func2a:
	.vsave	{d16}
	vpush	{d16}
	vpop	{d16}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func2b
	.align	2
	.type	func2b,%function
	.fnstart
func2b:
	.vsave	{d16, d17, d18, d19}
	vpush	{d16, d17, d18, d19}
	vpop	{d16, d17, d18, d19}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func2c
	.align	2
	.type	func2c,%function
	.fnstart
func2c:
	.vsave	{d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
	vpush	{d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
	vpop	{d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST2
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B000C800 00000000 B003C800  |................|
@ CHECK:     0010: 00000000 B00FC800                    |........|
@ CHECK:   )
@ CHECK: }
