@ RUN: llvm-mc -triple armv7-linux-eabi -filetype obj %s -o %t
@ RUN: llvm-readobj -u %t | FileCheck --check-prefixes=CHECK,SYM %s

@@ If .symtab doesn't exist, we can still dump some information.
@ RUN: llvm-objcopy --allow-broken-links --strip-all %t - | llvm-readobj -u - | FileCheck %s

	.syntax unified

	.cpu cortex-a8
	.fpu neon

	.section .personality

	.type __personality,%function
__personality:
	.fnstart
	bkpt
	.fnend


	.section .personality0

	.type personality0,%function
personality0:
	.fnstart
	bx lr
	.fnend


	.section .personality1

	.type personality1,%function
personality1:
	.fnstart
	.pad #0x100
	sub sp, sp, #0x100
	.save {r0-r11}
	push {r0-r11}
	pop {r0-r11}
	add sp, sp, #0x100
	bx lr
	.fnend


	.section .custom_personality

	.type custom_personality,%function
custom_personality:
	.fnstart
	.personality __personality
	bx lr
	.fnend


	.section .opcodes

	.type opcodes,%function
opcodes:
	.fnstart
	.vsave {d8-d12}
	vpush {d8-d12}
	vpop {d8-d12}
	bx lr
	.fnend


	.section .multiple

	.type function0,%function
function0:
	.fnstart
	bx lr
	.fnend

	.type function1,%function
function1:
	.fnstart
	.personality __personality
	bx lr
	.fnend

	.type function2,%function
function2:
	.fnstart
	bx lr
	.fnend

	.section .raw

	.type raw,%function
	.thumb_func
raw:
	.fnstart
	.unwind_raw 12, 0x02
	.unwind_raw -12, 0x42
	.unwind_raw 0, 0x80, 0x00
	.unwind_raw 4, 0x81, 0x00
	.unwind_raw 4, 0x80, 0x01
	.unwind_raw 8, 0x80, 0xc0
	.unwind_raw 12, 0x84, 0xc0
	.unwind_raw 0, 0x91
	.unwind_raw 8, 0xa1
	.unwind_raw 12, 0xa9
	.unwind_raw 0, 0xb0
	.unwind_raw 4, 0xb1, 0x01
	.unwind_raw 0xa04, 0xb2, 0x80, 0x04
	.unwind_raw 24, 0xb3, 0x12
	.unwind_raw 24, 0xba
	.unwind_raw 24, 0xc2
	.unwind_raw 24, 0xc6, 0x02
	.unwind_raw 8, 0xc7, 0x03
	.unwind_raw 24, 0xc8, 0x02
	.unwind_raw 24, 0xc9, 0x02
	.unwind_raw 64, 0xd7
	.fnend

	.section .spare

	.type spare,%function
spare:
	.fnstart
	.unwind_raw 4, 0x00
	.unwind_raw -4, 0x40
	.unwind_raw 0, 0x80, 0x00
	.unwind_raw 4, 0x88, 0x00
	.unwind_raw 0, 0x91
	.unwind_raw 0, 0x9d
	.unwind_raw 0, 0x9f
	.unwind_raw 0, 0xa0
	.unwind_raw 0, 0xa8
	.unwind_raw 0, 0xb0
	.unwind_raw 0, 0xb1, 0x00
	.unwind_raw 4, 0xb1, 0x01
	.unwind_raw 0, 0xb1, 0x10
	.unwind_raw 0x204, 0xb2, 0x00
	.unwind_raw 16, 0xb3, 0x00
	.unwind_raw 0, 0xb4
	.unwind_raw 16, 0xb8
	.unwind_raw 4, 0xc0
	.unwind_raw 4, 0xc6, 0x00
	.unwind_raw 4, 0xc7, 0x00
	.unwind_raw 4, 0xc7, 0x01
	.unwind_raw 0, 0xc7, 0x10
	.unwind_raw 16, 0xc8, 0x00
	.unwind_raw 16, 0xc9, 0x00
	.unwind_raw 0, 0xca
	.unwind_raw 16, 0xd0
	.unwind_raw 0, 0xd8
	.fnend

@ CHECK: UnwindInformation {
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.personality
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ SYM:           FunctionName: __personality
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         Opcodes [
@ CHECK:           0xB0       ; finish
@ CHECK:           0xB0       ; finish
@ CHECK:           0xB0       ; finish
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.personality0
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ SYM:           FunctionName: personality0
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         Opcodes [
@ CHECK:           0xB0       ; finish
@ CHECK:           0xB0       ; finish
@ CHECK:           0xB0       ; finish
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.personality1
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ SYM:           FunctionName: personality1
@ SYM:           ExceptionHandlingTable: .ARM.extab.personality1
@ SYM:           TableEntryOffset: 0x0
@ SYM:           Model: Compact
@ SYM:           PersonalityIndex: 1
@ SYM:           Opcodes [
@ SYM:             0xB1 0x0F ; pop {r0, r1, r2, r3}
@ SYM:             0xA7      ; pop {r4, r5, r6, r7, r8, r9, r10, fp}
@ SYM:             0x3F      ; vsp = vsp + 256
@ SYM:             0xB0      ; finish
@ SYM:             0xB0      ; finish
@ SYM:           ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.custom_personality
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ SYM:           FunctionName: custom_personality
@ SYM:           ExceptionHandlingTable: .ARM.extab.custom_personality
@ SYM:           TableEntryOffset: 0x0
@ SYM:           Model: Generic
@ SYM:           PersonalityRoutineAddress: 0x0
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.opcodes
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ SYM:           FunctionName: opcodes
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         Opcodes [
@ CHECK:           0xC9 0x84 ; pop {d8, d9, d10, d11, d12}
@ CHECK:           0xB0      ; finish
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.multiple
@ CHECK:     Entries [
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x0
@ SYM:           FunctionName: function0
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         Opcodes [
@ CHECK:           0xB0     ; finish
@ CHECK:           0xB0     ; finish
@ CHECK:           0xB0     ; finish
@ CHECK:         ]
@ CHECK:       }
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x4
@ SYM:           FunctionName: function1
@ SYM:           ExceptionHandlingTable: .ARM.extab.multiple
@ SYM:           Model: Generic
@ SYM:           PersonalityRoutineAddress: 0x0
@ CHECK:       }
@ CHECK:       Entry {
@ CHECK:         FunctionAddress: 0x8
@ SYM:           FunctionName: function2
@ CHECK:         Model: Compact (Inline)
@ CHECK:         PersonalityIndex: 0
@ CHECK:         Opcodes [
@ CHECK:           0xB0     ; finish
@ CHECK:           0xB0     ; finish
@ CHECK:           0xB0     ; finish
@ CHECK:         ]
@ CHECK:       }
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.raw
@ CHECK:     Entries [
@ SYM:         Opcodes [
@ SYM:           0xD7      ; pop {d8, d9, d10, d11, d12, d13, d14, d15}
@ SYM:           0xC9 0x02 ; pop {d0, d1, d2}
@ SYM:           0xC8 0x02 ; pop {d16, d17, d18}
@ SYM:           0xC7 0x03 ; pop {wCGR0, wCGR1}
@ SYM:           0xC6 0x02 ; pop {wR0, wR1, wR2}
@ SYM:           0xC2      ; pop {wR10, wR11, wR12}
@ SYM:           0xBA      ; pop {d8, d9, d10}
@ SYM:           0xB3 0x12 ; pop {d1, d2, d3}
@ SYM:           0xB2 0x80 0x04 ; vsp = vsp + 2564
@ SYM:           0xB1 0x01 ; pop {r0}
@ SYM:           0xB0      ; finish
@ SYM:           0xA9      ; pop {r4, r5, lr}
@ SYM:           0xA1      ; pop {r4, r5}
@ SYM:           0x91      ; vsp = r1
@ SYM:           0x84 0xC0 ; pop {r10, fp, lr}
@ SYM:           0x80 0xC0 ; pop {r10, fp}
@ SYM:           0x80 0x01 ; pop {r4}
@ SYM:           0x81 0x00 ; pop {ip}
@ SYM:           0x80 0x00 ; refuse to unwind
@ SYM:           0x42      ; vsp = vsp - 12
@ SYM:           0x02      ; vsp = vsp + 12
@ SYM:         ]
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   UnwindIndexTable {
@ CHECK:     SectionName: .ARM.exidx.spare
@ CHECK:     Entries [
@ SYM:         Opcodes [
@ SYM:           0xD8      ; spare
@ SYM:           0xD0      ; pop {d8}
@ SYM:           0xCA      ; spare
@ SYM:           0xC9 0x00 ; pop {d0}
@ SYM:           0xC8 0x00 ; pop {d16}
@ SYM:           0xC7 0x10 ; spare
@ SYM:           0xC7 0x01 ; pop {wCGR0}
@ SYM:           0xC7 0x00 ; spare
@ SYM:           0xC6 0x00 ; pop {wR0}
@ SYM:           0xC0      ; pop {wR10}
@ SYM:           0xB8      ; pop {d8}
@ SYM:           0xB4      ; spare
@ SYM:           0xB3 0x00 ; pop {d0}
@ SYM:           0xB2 0x00 ; vsp = vsp + 516
@ SYM:           0xB1 0x10 ; spare
@ SYM:           0xB1 0x01 ; pop {r0}
@ SYM:           0xB1 0x00 ; spare
@ SYM:           0xB0      ; finish
@ SYM:           0xA8      ; pop {r4, lr}
@ SYM:           0xA0      ; pop {r4}
@ SYM:           0x9F      ; reserved (WiMMX MOVrr)
@ SYM:           0x9D      ; reserved (ARM MOVrr)
@ SYM:           0x91      ; vsp = r1
@ SYM:           0x88 0x00 ; pop {pc}
@ SYM:           0x80 0x00 ; refuse to unwind
@ SYM:           0x40      ; vsp = vsp - 4
@ SYM:           0x00      ; vsp = vsp + 4
@ SYM:         ]
@ CHECK:     ]
@ CHECK:   }
@ CHECK: }

