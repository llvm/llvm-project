@ Test ARM / Thumb mode switching with .code
@ RUN: llvm-mc -triple armv7-unknown-unknown -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple thumbv7-unknown-unknown -show-encoding <%s | FileCheck %s

.code 16
	add.w	r0, r0, r1
@ CHECK: add.w	r0, r0, r1              @ encoding: [0x00,0xeb,0x01,0x00]

.code 32
	add	r0, r0, r1
@ CHECK: add	r0, r0, r1              @ encoding: [0x01,0x00,0x80,0xe0]

.code 16
        adds    r0, r0, r1
@ CHECK: adds	r0, r0, r1              @ encoding: [0x40,0x18]

.arm
	add	r0, r0, r1
@ CHECK: add	r0, r0, r1              @ encoding: [0x01,0x00,0x80,0xe0]

.thumb
	add.w	r0, r0, r1
        adds    r0, r0, r1

@ CHECK: add.w	r0, r0, r1              @ encoding: [0x00,0xeb,0x01,0x00]
@ CHECK: adds	r0, r0, r1              @ encoding: [0x40,0x18]
