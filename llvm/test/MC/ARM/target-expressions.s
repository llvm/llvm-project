@ RUN: llvm-mc -triple armv7-eabi -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:   | FileCheck -check-prefix CHECK-RELOCATIONS %s

	.syntax unified

	.type function,%function
function:
	bx lr

	.global external
	.type external,%function

.set deadbeat, 0xdeadbea7

	.type test,%function
test:
	movw r0, :lower16:function
	movt r0, :upper16:function

	movw r1, #:lower16:function
	movt r1, #:upper16:function

	movw r2, :lower16:deadbeat
	movt r2, :upper16:deadbeat

	movw r3, #:lower16:deadbeat
	movt r3, #:upper16:deadbeat

	movw r4, :lower16:0xD1510D6E
	movt r4, :upper16:0xD1510D6E

	movw r5, #:lower16:0xD1510D6E
	movt r5, #:upper16:0xD1510D6E

	movw r0, :lower16:external
	movt r0, :upper16:external

	movw r1, #:lower16:external
	movt r1, #:upper16:external

	movw r2, #:lower16:(16 + 16)
	movt r2, #:upper16:(16 + 16)

	movw r3, :lower16:(16 + 16)
	movt r3, :upper16:(16 + 16)

@ CHECK-LABEL: test:
@ CHECK: 	movw r0, :lower16:function
@ CHECK: 	movt r0, :upper16:function
@ CHECK: 	movw r1, :lower16:function
@ CHECK: 	movt r1, :upper16:function
@ CHECK: 	movw r2, :lower16:(3735928487)
@ CHECK: 	movt r2, :upper16:(3735928487)
@ CHECK: 	movw r3, :lower16:(3735928487)
@ CHECK: 	movt r3, :upper16:(3735928487)
@ CHECK: 	movw r4, :lower16:(3511749998)
@ CHECK: 	movt r4, :upper16:(3511749998)
@ CHECK: 	movw r5, :lower16:(3511749998)
@ CHECK: 	movt r5, :upper16:(3511749998)
@ CHECK: 	movw r0, :lower16:external
@ CHECK: 	movt r0, :upper16:external
@ CHECK: 	movw r1, :lower16:external
@ CHECK: 	movt r1, :upper16:external
@ CHECK: 	movw r2, :lower16:(32)
@ CHECK: 	movt r2, :upper16:(32)
@ CHECK: 	movw r3, :lower16:(32)
@ CHECK: 	movt r3, :upper16:(32)

@ CHECK-RELOCATIONS: Relocations [
@ CHECK-RELOCATIONS:   0x4 R_ARM_MOVW_ABS_NC function 0x0
@ CHECK-RELOCATIONS:   0x8 R_ARM_MOVT_ABS function 0x0
@ CHECK-RELOCATIONS:   0xC R_ARM_MOVW_ABS_NC function 0x0
@ CHECK-RELOCATIONS:   0x10 R_ARM_MOVT_ABS function 0x0
@ CHECK-RELOCATIONS:   0x34 R_ARM_MOVW_ABS_NC external 0x0
@ CHECK-RELOCATIONS:   0x38 R_ARM_MOVT_ABS external 0x0
@ CHECK-RELOCATIONS:   0x3C R_ARM_MOVW_ABS_NC external 0x0
@ CHECK-RELOCATIONS:   0x40 R_ARM_MOVT_ABS external 0x0
@ CHECK-RELOCATIONS: ]

