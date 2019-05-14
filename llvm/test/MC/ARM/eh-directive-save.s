@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd | FileCheck %s

@ Check the .save directive

@ The .save directive records the GPR registers which are pushed to the
@ stack.  There are 4 different unwind opcodes:
@
@     0xB100: pop r[3:0]
@     0xA0:   pop r[(4+x):4]		@ r[4+x]-r[4] must be consecutive.
@     0xA8:   pop r14, r[(4+x):4]	@ r[4+x]-r[4] must be consecutive.
@     0x8000: pop r[15:4]
@
@ If register list specifed by .save directive is possible to be encoded
@ by 0xA0 or 0xA8, then the assembler should prefer them over 0x8000.


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
	.save	{r0}
	push	{r0}
	pop	{r0}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func1b
	.align	2
	.type	func1b,%function
	.fnstart
func1b:
	.save	{r0, r1}
	push	{r0, r1}
	pop	{r0, r1}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func1c
	.align	2
	.type	func1c,%function
	.fnstart
func1c:
	.save	{r0, r2}
	push	{r0, r2}
	pop	{r0, r2}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func1d
	.align	2
	.type	func1d,%function
	.fnstart
func1d:
	.save	{r1, r2}
	push	{r1, r2}
	pop	{r1, r2}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func1e
	.align	2
	.type	func1e,%function
	.fnstart
func1e:
	.save	{r0, r1, r2, r3}
	push	{r0, r1, r2, r3}
	pop	{r0, r1, r2, r3}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0xB000 unwind opcode.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST1
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B001B100 00000000 B003B100  |................|
@ CHECK:     0010: 00000000 B005B100 00000000 B006B100  |................|
@ CHECK:     0020: 00000000 B00FB100                    |........|
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
	.save	{r4}
	push	{r4}
	pop	{r4}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func2b
	.align	2
	.type	func2b,%function
	.fnstart
func2b:
	.save	{r4, r5}
	push	{r4, r5}
	pop	{r4, r5}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func2c
	.align	2
	.type	func2c,%function
	.fnstart
func2c:
	.save	{r4, r5, r6, r7, r8, r9, r10, r11}
	push	{r4, r5, r6, r7, r8, r9, r10, r11}
	pop	{r4, r5, r6, r7, r8, r9, r10, r11}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0xA0 unwind opcode.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST2
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B0A000 00000000 B0B0A100  |................|
@ CHECK:     0010: 00000000 B0B0A700                    |........|
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
	.save	{r4, r14}
	push	{r4, r14}
	pop	{r4, r14}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func3b
	.align	2
	.type	func3b,%function
	.fnstart
func3b:
	.save	{r4, r5, r14}
	push	{r4, r5, r14}
	pop	{r4, r5, r14}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func3c
	.align	2
	.type	func3c,%function
	.fnstart
func3c:
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, r14}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, r14}
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, r14}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0xA8 unwind opcode.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST3
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B0A800 00000000 B0B0A900  |................|
@ CHECK:     0010: 00000000 B0B0AF00                    |........|
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
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, r12, r14}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, r12, r14}
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, r12, r14}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func4b
	.align	2
	.type	func4b,%function
	.fnstart
func4b:
	@ Note: r7 is missing intentionally.
	.save	{r4, r5, r6, r8, r9, r10, r11}
	push	{r4, r5, r6, r8, r9, r10, r11}
	pop	{r4, r5, r6, r8, r9, r10, r11}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func4c
	.align	2
	.type	func4c,%function
	.fnstart
func4c:
	@ Note: r7 is missing intentionally.
	.save	{r4, r5, r6, r8, r9, r10, r11, r14}
	push	{r4, r5, r6, r8, r9, r10, r11, r14}
	pop	{r4, r5, r6, r8, r9, r10, r11, r14}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func4d
	.align	2
	.type	func4d,%function
	.fnstart
func4d:
	@ Note: The register list is not start with r4.
	.save	{r5, r6, r7}
	push	{r5, r6, r7}
	pop	{r5, r6, r7}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func4e
	.align	2
	.type	func4e,%function
	.fnstart
func4e:
	@ Note: The register list is not start with r4.
	.save	{r5, r6, r7, r14}
	push	{r5, r6, r7, r14}
	pop	{r5, r6, r7, r14}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ The assembler should emit 0x8000 unwind opcode.
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST4
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0FF8500 00000000 B0F78000  |................|
@ CHECK:     0010: 00000000 B0F78400 00000000 B00E8000  |................|
@ CHECK:     0020: 00000000 B00E8400                    |........|
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
	.save	{r0, r1, r2, r3, r4, r5, r6}
	push	{r0, r1, r2, r3, r4, r5, r6}
	pop	{r0, r1, r2, r3, r4, r5, r6}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

	.globl	func5b
	.align	2
	.type	func5b,%function
	.fnstart
func5b:
	.save	{r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r14}
	push	{r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r14}
	pop	{r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r14}
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend

@-------------------------------------------------------------------------------
@ Check the order of unwind opcode to pop registers.
@ 0xB10F "pop {r0-r3}" should be emitted before 0xA2 "pop {r4-r6}".
@ 0xB10F "pop {r0-r3}" should be emitted before 0x85FF "pop {r4-r12, r14}".
@-------------------------------------------------------------------------------
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST5
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 A20FB100 00000000 850FB101  |................|
@ CHECK:     0010: B0B0B0FF                             |....|
@ CHECK:   )
@ CHECK: }
