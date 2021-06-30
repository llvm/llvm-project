@ RUN: llvm-mc -triple thumbv7-apple-darwin -filetype=obj -o %t.o %s
@ RUN: llvm-objdump --triple=thumbv7-apple-darwin -d %t.o | FileCheck %s

.thumb
start:
.thumb_func start
	add	 r1, r2, r3
	cbnz	 r2, L1 @ this can't be encoded, must turn into a nop
L1:
	add	r4, r5, r6
	cbnz	r2, L2
	sub	r7, r8, r9
L2:
	add	r7, r8, r9
	cbz	r2, L3 @ this can't be encoded, must turn into a nop
L3:
	add	r10, r11, r12
	cbz	r2, L4
	sub	r7, r8, r9
L4:
	add	r3, r4, r5

@ CHECK: 0:	02 eb 03 01	add.w	r1, r2, r3
@ CHECK: 4:	00 bf		nop
@ CHECK: 6:	05 eb 06 04	add.w	r4, r5, r6
@ CHECK: a:	0a b9		cbnz	r2, 0x10 <start+0x10> @ imm = #2
@ CHECK: c:	a8 eb 09 07	sub.w	r7, r8, r9
@ CHECK: 10:	08 eb 09 07	add.w	r7, r8, r9
@ CHECK: 14:	00 bf		nop
@ CHECK: 16:	0b eb 0c 0a	add.w	r10, r11, r12
@ CHECK: 1a:	0a b1		cbz	r2, 0x20 <start+0x20> @ imm = #2
@ CHECK: 1c:	a8 eb 09 07	sub.w	r7, r8, r9
@ CHECK: 20:	04 eb 05 03	add.w	r3, r4, r5
