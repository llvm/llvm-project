@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd | FileCheck %s

@ Check for .setfp directive.

@ The .setfp directive will track the offset between the frame pointer and
@ the stack pointer.  This is required for the function that will change
@ the stack pointer out of the function prologue.  If the exception is thrown,
@ then libunwind will reconstruct the stack pointer from the frame pointer.
@ The reconstruction code is implemented by two different unwind opcode:
@ (i) the unwind opcode to copy stack offset from the other register, and
@ (ii) the unwind opcode to add or subtract the stack offset.
@
@ This file includes several cases separated by different range of -offset
@
@              (-offset) <  0x00
@              (-offset) == 0x00
@     0x04  <= (-offset) <= 0x100
@     0x104 <= (-offset) <= 0x200
@     0x204 <= (-offset)


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
	.setfp	fp, sp, #0
	add	fp, sp, #0
	sub	sp, fp, #0
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0x9B to copy stack pointer from r11.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST1
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B09B00                    |........|
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
	.setfp	fp, sp, #-4
	add	fp, sp, #4
	sub	sp, fp, #4
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func2b
	.align	2
	.type	func2b,%function
	.fnstart
func2b:
	.setfp	fp, sp, #-0x100
	add	fp, sp, #0x100
	sub	sp, fp, #0x100
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0x9B to copy stack pointer from r11.
@ The assembler should emit ((-offset - 4) >> 2) for offset.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST2
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0009B00 00000000 B03F9B00  |.............?..|
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
	.setfp	fp, sp, #-0x104
	sub	fp, sp, #0x104
	add	sp, fp, #0x104
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func3b
	.align	2
	.type	func3b,%function
	.fnstart
func3b:
	.setfp	fp, sp, #-0x200
	sub	fp, sp, #0x200
	add	sp, fp, #0x200
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0x9B to copy stack pointer from r11.
@ The assembler should emit 0x3F and ((-offset - 0x104) >> 2) for offset.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST3
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 3F009B00 00000000 3F3F9B00  |....?.......??..|
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
	.setfp	fp, sp, #-0x204
	sub	fp, sp, #0x204
	add	sp, fp, #0x204
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func4b
	.align	2
	.type	func4b,%function
	.fnstart
func4b:
	.setfp	fp, sp, #-0x580
	sub	fp, sp, #0x580
	add	sp, fp, #0x580
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0x9B to copy stack pointer from r11.
@ The assembler should emit 0xB2 and the ULEB128 encoding of
@ ((-offset - 0x204) >> 2) for offset.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST4
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 00B29B00 00000000 DFB29B01  |................|
@ CHECK:     0010: B0B0B001                             |....|
@ CHECK:   )
@ CHECK: }



@-------------------------------------------------------------------------------
@ TEST5
@-------------------------------------------------------------------------------
	.section	.TEST5
	.globl	func5a
	.align	2
	.type	func5a,%function
	.fnstart
func5a:
	.setfp	fp, sp, #0x4
	add	fp, sp, #0x4
	sub	sp, fp, #0x4
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func5b
	.align	2
	.type	func5b,%function
	.fnstart
func5b:
	.setfp	fp, sp, #0x104
	add	fp, sp, #0x104
	sub	sp, fp, #0x104
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func5c
	.align	2
	.type	func5c,%function
	.fnstart
func5c:
	.setfp	fp, sp, #0x204
	add	fp, sp, #0x204
	sub	sp, fp, #0x204
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0x9B to copy stack pointer from r11.
@ The assembler should emit (0x40 | (offset - 4)) >> 2 for offset.
@ If (offset - 4) is greater than 0x3f, then multiple 0x7f should be emitted.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST5
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0409B00 00000000 7F409B00  |.....@.......@..|
@ CHECK:     0010: 00000000 7F409B01 B0B0B07F           |.....@......|
@ CHECK:   )
@ CHECK: }
