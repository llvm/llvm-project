@ RUN: not llvm-mc -triple thumbv7-eabi -filetype asm -o - %s 2>&1 \
@ RUN:     | FileCheck %s
@ RUN: not llvm-mc -triple thumbv7a-eabi -filetype asm -o - %s 2>&1 \
@ RUN: | FileCheck --check-prefix=CHECK --check-prefix=CHECK-V7A %s
@ RUN: not llvm-mc -triple thumbv7m-eabi -filetype asm -o - %s 2>&1 \
@ RUN: | FileCheck --check-prefix=CHECK --check-prefix=CHECK-V7M %s

	.syntax unified
	.thumb

	.global ldm
	.type ldm,%function
ldm:
	ldm r0!, {r1, sp}
@ CHECK: error: SP may not be in the register list
@ CHECK: ldm r0!, {r1, sp}
@ CHECK:          ^
	ldm r0!, {lr, pc}
@ CHECK: error: PC and LR may not be in the register list simultaneously
@ CHECK: ldm r0!, {lr, pc}
@ CHECK:          ^
	itt eq
	ldmeq r0!, {r1, pc}
	ldmeq r0!, {r2, lr}
@ CHECK: error: instruction must be outside of IT block or the last instruction in an IT block
@ CHECK: ldmeq r0!, {r1, pc}
@ CHECK:            ^

	.global ldmdb
	.type ldmdb,%function
ldmdb:
	ldmdb r0!, {r1, sp}
@ CHECK: error: SP may not be in the register list
	ldmdb r0!, {lr, pc}
@ error: PC and LR may not be in the register list simultaneously
	itt eq
	ldmeq r0!, {r1, pc}
	ldmeq r0!, {r2, lr}
@ CHECK: error: instruction must be outside of IT block or the last instruction in an IT block
@ CHECK: ldmeq r0!, {r1, pc}
@ CHECK:            ^

	.global stm
	.type stm,%function
stm:
	stm r0!, {r1, sp}
@ CHECK: error: SP may not be in the register list
	stm r0!, {r2, pc}
@ CHECK: error: PC may not be in the register list
	stm r0!, {sp, pc}
@ CHECK: error: SP and PC may not be in the register list

	.global stmdb
	.type stmdb,%function
stmdb:
	stmdb r0!, {r1, sp}
@ CHECK: error: SP may not be in the register list
	stmdb r0!, {r2, pc}
@ CHECK: error: PC may not be in the register list
	stmdb r0!, {sp, pc}
@ CHECK: error: SP and PC may not be in the register list

	.global push
	.type push,%function
push:
	push {sp}
@ CHECK: error: SP may not be in the register list
	push {pc}
@ CHECK: error: PC may not be in the register list
	push {sp, pc}
@ CHECK: error: SP and PC may not be in the register list

	.global pop
	.type pop,%function
pop:
        pop {sp}
@ CHECK-V7M: error: SP may not be in the register list
	pop {lr, pc}
@ CHECK: error: PC and LR may not be in the register list simultaneously
@ CHECK: pop {lr, pc}
@ CHECK:     ^
	itt eq
	popeq {r1, pc}
	popeq {r2, lr}
@ CHECK: error: instruction must be outside of IT block or the last instruction in an IT block
@ CHECK: popeq {r1, pc}
@ CHECK:     ^

	.global valid
	.type valid,%function
valid:
	pop {sp}
@ CHECK-V7A: ldr sp, [sp], #4
	pop {sp, pc}
@ CHECK-V7A: pop.w {sp, pc}
	push.w {r0}
@ CHECK: str r0, [sp, #-4]
	pop.w {r0}
@ CHECK: ldr r0, [sp], #4

