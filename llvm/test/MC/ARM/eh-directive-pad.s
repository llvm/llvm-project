@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd - | FileCheck %s

@ Check for different stack pointer offsets.

@ The .pad directive will track the stack pointer offsets.  There are several
@ ways to encode the stack offsets.  We have to test:
@
@              offset <  0x00
@              offset == 0x00
@     0x04  <= offset <= 0x100
@     0x104 <= offset <= 0x200
@     0x204 <= offset


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
	.pad	#0
	sub	sp, sp, #0
	add	sp, sp, #0
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit nothing (will be filled up with finish opcode).
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST1
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B0B000                    |........|
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
	.pad	#0x4
	sub	sp, sp, #0x4
	add	sp, sp, #0x4
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func2b
	.align	2
	.type	func2b,%function
	.fnstart
func2b:
	.pad	#0x100
	sub	sp, sp, #0x100
	add	sp, sp, #0x100
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit ((offset - 4) >> 2).
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST2
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B00000 00000000 B0B03F00  |..............?.|
@ CHECK:   )
@ CHECK: }


@-------------------------------------------------------------------------------
@ TEST3
@-------------------------------------------------------------------------------
	.section	.TEST3
	.globl	func3a
	.align	2
	.type	func3a,%function
	.fnstart
func3a:
	.pad	#0x104
	sub	sp, sp, #0x104
	add	sp, sp, #0x104
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func3b
	.align	2
	.type	func3b,%function
	.fnstart
func3b:
	.pad	#0x200
	sub	sp, sp, #0x200
	add	sp, sp, #0x200
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0x3F and ((offset - 0x104) >> 2).
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST3
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B03F0000 00000000 B03F3F00  |.....?.......??.|
@ CHECK:   )
@ CHECK: }



@-------------------------------------------------------------------------------
@ TEST4
@-------------------------------------------------------------------------------
	.section	.TEST4
	.globl	func4a
	.align	2
	.type	func4a,%function
	.fnstart
func4a:
	.pad	#0x204
	sub	sp, sp, #0x204
	add	sp, sp, #0x204
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func4b
	.align	2
	.type	func4b,%function
	.fnstart
func4b:
	.pad	#0x580
	sub	sp, sp, #0x580
	add	sp, sp, #0x580
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0xB2 and the ULEB128 encoding of
@ ((offset - 0x204) >> 2).
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST4
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B000B200 00000000 01DFB200  |................|
@ CHECK:   )
@ CHECK: }



@-------------------------------------------------------------------------------
@ TEST5
@-------------------------------------------------------------------------------
	.section	.TEST5
	.globl	func4a
	.align	2
	.type	func4a,%function
	.fnstart
func5a:
	.pad	#-0x4
	add	sp, sp, #0x4
	sub	sp, sp, #0x4
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func5b
	.align	2
	.type	func5b,%function
	.fnstart
func5b:
	.pad	#-0x104
	add	sp, sp, #0x104
	sub	sp, sp, #0x4
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func5c
	.align	2
	.type	func5c,%function
	.fnstart
func5c:
	.pad	#-0x204
	add	sp, sp, #0x204
	sub	sp, sp, #0x4
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit (0x40 | (-offset - 4)) >> 2.  When (-offset - 4)
@ is greater than 0x3f, then multiple 0x7f should be emitted.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST5
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B04000 00000000 B07F4000  |......@.......@.|
@ CHECK:     0010: 00000000 7F7F4000                    |......@.|
@ CHECK:   )
@ CHECK: }
