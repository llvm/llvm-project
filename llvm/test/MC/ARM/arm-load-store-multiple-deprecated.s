@ RUN: llvm-mc -triple armv6t2-linux-eabi -filetype asm -o - %s 2>&1 \
@ RUN:   | FileCheck %s
@ RUN: not llvm-mc -triple armv6t2-linux-eabi --fatal-warnings -filetype asm -o - %s 2>&1 \
@ RUN:   | FileCheck -check-prefix CHECK-ERROR %s

@ RUN: not llvm-mc -triple armv7-linux-eabi -filetype asm -o - %s 2>&1 \
@ RUN:   | FileCheck %s -check-prefix CHECK -check-prefix CHECK-V7

	.syntax unified
	.arm

	.global stm
	.type stm,%function
stm:
	stm sp!, {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK-ERROR: error: use of SP or PC in the list is deprecated
@ CHECK: stm sp!, {r0, pc}
@ CHECK: ^
	stm r0!, {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK-ERROR: error: use of SP or PC in the list is deprecated
@ CHECK: stm r0!, {r0, sp}
@ CHECK: ^
	stm r1!, {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm r1!, {r0, sp, pc}
@ CHECK: ^
	stm r2!, {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm r2!, {sp, pc}
@ CHECK: ^
	stm sp!, {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm sp!, {pc}
@ CHECK: ^
	stm r0!, {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stm r0!, {sp}
@ CHECK: ^

	.global stmda
	.type stmda,%function
stmda:
	stmda sp!, {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda sp!, {r0, pc}
@ CHECK: ^
	stmda r0!, {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda r0!, {r0, sp}
@ CHECK: ^
	stmda r1!, {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda r1!, {r0, sp, pc}
@ CHECK: ^
	stmda r2!, {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda r2!, {sp, pc}
@ CHECK: ^
	stmda sp!, {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda sp!, {pc}
@ CHECK: ^
	stmda r0!, {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmda r0!, {sp}
@ CHECK: ^

	.global stmdb
	.type stmdb,%function
stmdb:
	stmdb sp!, {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb sp!, {r0, pc}
@ CHECK: ^
	stmdb r0!, {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb r0!, {r0, sp}
@ CHECK: ^
	stmdb r1!, {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb r1!, {r0, sp, pc}
@ CHECK: ^
	stmdb r2!, {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb r2!, {sp, pc}
@ CHECK: ^
	stmdb sp!, {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb sp!, {pc}
@ CHECK: ^
	stmdb r0!, {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmdb r0!, {sp}
@ CHECK: ^

	.global stmib
	.type stmib,%function
stmib:
	stmib sp!, {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib sp!, {r0, pc}
@ CHECK: ^
	stmib r0!, {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib r0!, {r0, sp}
@ CHECK: ^
	stmib r1!, {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib r1!, {r0, sp, pc}
@ CHECK: ^
	stmib r2!, {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib r2!, {sp, pc}
@ CHECK: ^
	stmib sp!, {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib sp!, {pc}
@ CHECK: ^
	stmib r0!, {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: stmib r0!, {sp}
@ CHECK: ^


	.global push
	.type push,%function
push:
	push {r0, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {r0, pc}
@ CHECK: ^
	push {r0, sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {r0, sp}
@ CHECK: ^
	push {r0, sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {r0, sp, pc}
@ CHECK: ^
	push {sp, pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {sp, pc}
@ CHECK: ^
	push {pc}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {pc}
@ CHECK: ^
	push {sp}
@ CHECK: warning: use of SP or PC in the list is deprecated
@ CHECK: push {sp}
@ CHECK: ^

	.global ldm
	.type ldm,%function
ldm:
	ldm r0!, {r1, sp}
@ CHECK: warning: use of SP in the list is deprecated
	ldm r0!, {sp}
@ CHECK: warning: use of SP in the list is deprecated
	ldm r0!, {r1, lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated
	ldm r0!, {lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated

	.global ldmda
	.type ldmda,%function
ldmda:
	ldmda r0!, {r1, sp}
@ CHECK: warning: use of SP in the list is deprecated
	ldmda r0!, {sp}
@ CHECK: warning: use of SP in the list is deprecated
	ldmda r0!, {r1, lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated
	ldmda r0!, {lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated

	.global ldmdb
	.type ldmdb,%function
ldmdb:
	ldmdb r0!, {r1, sp}
@ CHECK: warning: use of SP in the list is deprecated
	ldmdb r0!, {sp}
@ CHECK: warning: use of SP in the list is deprecated
	ldmdb r0!, {r1, lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated
	ldmdb r0!, {lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated

	.global ldmib
	.type ldmib,%function
ldmib:
	ldmib r0!, {r1, sp}
@ CHECK: warning: use of SP in the list is deprecated
	ldmib r0!, {sp}
@ CHECK: warning: use of SP in the list is deprecated
	ldmib r0!, {r1, lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated
	ldmib r0!, {lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated

	.global pop
	.type pop,%function
pop:
	pop {r0, sp}
@ CHECK: warning: use of SP in the list is deprecated
@ CHECK-V7: error: writeback register not allowed in register list
	pop {sp}
@ CHECK: warning: use of SP in the list is deprecated
@ CHECK-V7: error: writeback register not allowed in register list
	pop {r0, lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated
	pop {lr, pc}
@ CHECK: warning: use of LR and PC simultaneously in the list is deprecated

	.global valid
	.type valid,%function
valid:
	stmdaeq r0, {r0}
@ CHECK: stmdaeq r0, {r0}
	ldmdaeq r0, {r0}
@ CHECK: ldmdaeq r0, {r0}
	pop {r0, pc}
@ CHECK: pop {r0, pc}

