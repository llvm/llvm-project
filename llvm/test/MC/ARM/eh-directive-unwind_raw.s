@ RUN: llvm-mc -triple armv7-linux-eabi -filetype obj -o - %s | llvm-readobj -u - \
@ RUN:   | FileCheck %s

	.syntax unified

	.type save,%function
	.thumb_func
save:
	.fnstart
	.unwind_raw 4, 0xb1, 0x01
	push {r0}
	pop {r0}
	bx lr
	.fnend

	.type empty,%function
	.thumb_func
empty:
	.fnstart
	.unwind_raw 0, 0xb0
	bx lr
	.fnend

	.type extended,%function
	.thumb_func
extended:
	.fnstart
	.unwind_raw 12, 0x9b, 0x40, 0x84, 0x80, 0xb0, 0xb0
	@ .save {fp, lr}
	stmfd sp!, {fp, lr}
	@ .setfp fp, sp, #4
	add fp, sp, #4
	@ .pad #8
	sub sp, sp, #8
	add sp, sp, #8
	sub fp, sp, #4
	ldmfd sp!, {fp, lr}
	bx lr
	.fnend

	.type refuse,%function
	.thumb_func
refuse:
	.fnstart
	.unwind_raw 0, 0x80, 0x00
	bx lr
	.fnend

	.type stack_adjust,%function
	.thumb_func
stack_adjust:
	.fnstart
	.setfp fp, sp, #32
	.unwind_raw 24, 0xc2
	.fnend

@ CHECK: UnwindInformation {
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         Opcodes [
@ CHECK:           0xB1 0x01 ; pop {r0}
@ CHECK:           0xB0      ; finish
@ CHECK:         ]
@ CHECK:       }
@ CHECK:       Entry {
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         Opcodes [
@ CHECK:           0xB0      ; finish
@ CHECK:           0xB0      ; finish
@ CHECK:           0xB0      ; finish
@ CHECK:         ]
@ CHECK:       }
@ CHECK:       Entry {
@ CHECK:         ExceptionHandlingTable: .ARM.extab
@ CHECK:         Model: Compact
@ CHECK:         PersonalityIndex: 1
@ CHECK:         Opcodes [
@ CHECK:           0x9B      ; vsp = r11
@ CHECK:           0x40      ; vsp = vsp - 4
@ CHECK:           0x84 0x80 ; pop {fp, lr}
@ CHECK:           0xB0      ; finish
@ CHECK:           0xB0      ; finish
@ CHECK:         ]
@ CHECK:       }
@ CHECK:       Entry {
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         Opcodes [
@ CHECK:           0x80 0x00 ; refuse to unwind
@ CHECK:           0xB0      ; finish
@ CHECK:         ]
@ CHECK:       }
@ CHECK:       Entry {
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         Opcodes [
@ CHECK:           0x9B      ; vsp = r11
@ CHECK:           0x4D      ; vsp = vsp - 56
@ CHECK:           0xC2      ; pop {wR10, wR11, wR12}
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK: }

