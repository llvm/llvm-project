# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: lr	%r0, %r1                # encoding: [0x18,0x01]
#CHECK: lr	%r2, %r3                # encoding: [0x18,0x23]
#CHECK: lr	%r4, %r5                # encoding: [0x18,0x45]
#CHECK: lr	%r6, %r7                # encoding: [0x18,0x67]
#CHECK: lr	%r8, %r9                # encoding: [0x18,0x89]
#CHECK: lr	%r10, %r11              # encoding: [0x18,0xab]
#CHECK: lr	%r12, %r13              # encoding: [0x18,0xcd]
#CHECK: lr	%r14, %r15              # encoding: [0x18,0xef]

	lr	%r0,%r1
	lr	%r2,%r3
	lr	%r4,%r5
	lr	%r6,%r7
	lr	%r8,%r9
	lr	%r10,%r11
	lr	%r12,%r13
	lr	%r14,%r15

#CHECK: lgr	%r0, %r1                # encoding: [0xb9,0x04,0x00,0x01]
#CHECK: lgr	%r2, %r3                # encoding: [0xb9,0x04,0x00,0x23]
#CHECK: lgr	%r4, %r5                # encoding: [0xb9,0x04,0x00,0x45]
#CHECK: lgr	%r6, %r7                # encoding: [0xb9,0x04,0x00,0x67]
#CHECK: lgr	%r8, %r9                # encoding: [0xb9,0x04,0x00,0x89]
#CHECK: lgr	%r10, %r11              # encoding: [0xb9,0x04,0x00,0xab]
#CHECK: lgr	%r12, %r13              # encoding: [0xb9,0x04,0x00,0xcd]
#CHECK: lgr	%r14, %r15              # encoding: [0xb9,0x04,0x00,0xef]

	lgr	%r0,%r1
	lgr	%r2,%r3
	lgr	%r4,%r5
	lgr	%r6,%r7
	lgr	%r8,%r9
	lgr	%r10,%r11
	lgr	%r12,%r13
	lgr	%r14,%r15

#CHECK: dlr	%r0, %r0                # encoding: [0xb9,0x97,0x00,0x00]
#CHECK: dlr	%r2, %r0                # encoding: [0xb9,0x97,0x00,0x20]
#CHECK: dlr	%r4, %r0                # encoding: [0xb9,0x97,0x00,0x40]
#CHECK: dlr	%r6, %r0                # encoding: [0xb9,0x97,0x00,0x60]
#CHECK: dlr	%r8, %r0                # encoding: [0xb9,0x97,0x00,0x80]
#CHECK: dlr	%r10, %r0               # encoding: [0xb9,0x97,0x00,0xa0]
#CHECK: dlr	%r12, %r0               # encoding: [0xb9,0x97,0x00,0xc0]
#CHECK: dlr	%r14, %r0               # encoding: [0xb9,0x97,0x00,0xe0]

	dlr	%r0,%r0
	dlr	%r2,%r0
	dlr	%r4,%r0
	dlr	%r6,%r0
	dlr	%r8,%r0
	dlr	%r10,%r0
	dlr	%r12,%r0
	dlr	%r14,%r0

#CHECK: ler	%f0, %f1                # encoding: [0x38,0x01]
#CHECK: ler	%f2, %f3                # encoding: [0x38,0x23]
#CHECK: ler	%f4, %f5                # encoding: [0x38,0x45]
#CHECK: ler	%f6, %f7                # encoding: [0x38,0x67]
#CHECK: ler	%f8, %f9                # encoding: [0x38,0x89]
#CHECK: ler	%f10, %f11              # encoding: [0x38,0xab]
#CHECK: ler	%f12, %f13              # encoding: [0x38,0xcd]
#CHECK: ler	%f14, %f15              # encoding: [0x38,0xef]

	ler	%f0,%f1
	ler	%f2,%f3
	ler	%f4,%f5
	ler	%f6,%f7
	ler	%f8,%f9
	ler	%f10,%f11
	ler	%f12,%f13
	ler	%f14,%f15

#CHECK: ldr	%f0, %f1                # encoding: [0x28,0x01]
#CHECK: ldr	%f2, %f3                # encoding: [0x28,0x23]
#CHECK: ldr	%f4, %f5                # encoding: [0x28,0x45]
#CHECK: ldr	%f6, %f7                # encoding: [0x28,0x67]
#CHECK: ldr	%f8, %f9                # encoding: [0x28,0x89]
#CHECK: ldr	%f10, %f11              # encoding: [0x28,0xab]
#CHECK: ldr	%f12, %f13              # encoding: [0x28,0xcd]
#CHECK: ldr	%f14, %f15              # encoding: [0x28,0xef]

	ldr	%f0,%f1
	ldr	%f2,%f3
	ldr	%f4,%f5
	ldr	%f6,%f7
	ldr	%f8,%f9
	ldr	%f10,%f11
	ldr	%f12,%f13
	ldr	%f14,%f15

#CHECK: lxr	%f0, %f1                # encoding: [0xb3,0x65,0x00,0x01]
#CHECK: lxr	%f4, %f5                # encoding: [0xb3,0x65,0x00,0x45]
#CHECK: lxr	%f8, %f9                # encoding: [0xb3,0x65,0x00,0x89]
#CHECK: lxr	%f12, %f13              # encoding: [0xb3,0x65,0x00,0xcd]

	lxr	%f0,%f1
	lxr	%f4,%f5
	lxr	%f8,%f9
	lxr	%f12,%f13

#CHECK: cpya	%a0, %a1                # encoding: [0xb2,0x4d,0x00,0x01]
#CHECK: cpya	%a2, %a3                # encoding: [0xb2,0x4d,0x00,0x23]
#CHECK: cpya	%a4, %a5                # encoding: [0xb2,0x4d,0x00,0x45]
#CHECK: cpya	%a6, %a7                # encoding: [0xb2,0x4d,0x00,0x67]
#CHECK: cpya	%a8, %a9                # encoding: [0xb2,0x4d,0x00,0x89]
#CHECK: cpya	%a10, %a11              # encoding: [0xb2,0x4d,0x00,0xab]
#CHECK: cpya	%a12, %a13              # encoding: [0xb2,0x4d,0x00,0xcd]
#CHECK: cpya	%a14, %a15              # encoding: [0xb2,0x4d,0x00,0xef]

	cpya	%a0,%a1
	cpya	%a2,%a3
	cpya	%a4,%a5
	cpya	%a6,%a7
	cpya	%a8,%a9
	cpya	%a10,%a11
	cpya	%a12,%a13
	cpya	%a14,%a15

#CHECK: lctl	%c0, %c1, 0             # encoding: [0xb7,0x01,0x00,0x00]
#CHECK: lctl	%c2, %c3, 0             # encoding: [0xb7,0x23,0x00,0x00]
#CHECK: lctl	%c4, %c5, 0             # encoding: [0xb7,0x45,0x00,0x00]
#CHECK: lctl	%c6, %c7, 0             # encoding: [0xb7,0x67,0x00,0x00]
#CHECK: lctl	%c8, %c9, 0             # encoding: [0xb7,0x89,0x00,0x00]
#CHECK: lctl	%c10, %c11, 0           # encoding: [0xb7,0xab,0x00,0x00]
#CHECK: lctl	%c12, %c13, 0           # encoding: [0xb7,0xcd,0x00,0x00]
#CHECK: lctl	%c14, %c15, 0           # encoding: [0xb7,0xef,0x00,0x00]

	lctl	%c0,%c1,0
	lctl	%c2,%c3,0
	lctl	%c4,%c5,0
	lctl	%c6,%c7,0
	lctl	%c8,%c9,0
	lctl	%c10,%c11,0
	lctl	%c12,%c13,0
	lctl	%c14,%c15,0


#CHECK: .cfi_offset %r0, 0
#CHECK: .cfi_offset %r1, 8
#CHECK: .cfi_offset %r2, 16
#CHECK: .cfi_offset %r3, 24
#CHECK: .cfi_offset %r4, 32
#CHECK: .cfi_offset %r5, 40
#CHECK: .cfi_offset %r6, 48
#CHECK: .cfi_offset %r7, 56
#CHECK: .cfi_offset %r8, 64
#CHECK: .cfi_offset %r9, 72
#CHECK: .cfi_offset %r10, 80
#CHECK: .cfi_offset %r11, 88
#CHECK: .cfi_offset %r12, 96
#CHECK: .cfi_offset %r13, 104
#CHECK: .cfi_offset %r14, 112
#CHECK: .cfi_offset %r15, 120
#CHECK: .cfi_offset %f0, 128
#CHECK: .cfi_offset %f1, 136
#CHECK: .cfi_offset %f2, 144
#CHECK: .cfi_offset %f3, 152
#CHECK: .cfi_offset %f4, 160
#CHECK: .cfi_offset %f5, 168
#CHECK: .cfi_offset %f6, 176
#CHECK: .cfi_offset %f7, 184
#CHECK: .cfi_offset %f8, 192
#CHECK: .cfi_offset %f9, 200
#CHECK: .cfi_offset %f10, 208
#CHECK: .cfi_offset %f11, 216
#CHECK: .cfi_offset %f12, 224
#CHECK: .cfi_offset %f13, 232
#CHECK: .cfi_offset %f14, 240
#CHECK: .cfi_offset %f15, 248
#CHECK: .cfi_offset %a0, 256
#CHECK: .cfi_offset %a1, 260
#CHECK: .cfi_offset %a2, 264
#CHECK: .cfi_offset %a3, 268
#CHECK: .cfi_offset %a4, 272
#CHECK: .cfi_offset %a5, 276
#CHECK: .cfi_offset %a6, 280
#CHECK: .cfi_offset %a7, 284
#CHECK: .cfi_offset %a8, 288
#CHECK: .cfi_offset %r9, 292
#CHECK: .cfi_offset %a10, 296
#CHECK: .cfi_offset %a11, 300
#CHECK: .cfi_offset %a12, 304
#CHECK: .cfi_offset %a13, 308
#CHECK: .cfi_offset %a14, 312
#CHECK: .cfi_offset %a15, 316
#CHECK: .cfi_offset %c0, 318
#CHECK: .cfi_offset %c1, 326
#CHECK: .cfi_offset %c2, 334
#CHECK: .cfi_offset %c3, 342
#CHECK: .cfi_offset %c4, 350
#CHECK: .cfi_offset %c5, 358
#CHECK: .cfi_offset %c6, 366
#CHECK: .cfi_offset %c7, 374
#CHECK: .cfi_offset %c8, 382
#CHECK: .cfi_offset %c9, 390
#CHECK: .cfi_offset %c10, 398
#CHECK: .cfi_offset %c11, 406
#CHECK: .cfi_offset %c12, 414
#CHECK: .cfi_offset %c13, 422
#CHECK: .cfi_offset %c14, 430
#CHECK: .cfi_offset %c15, 438

	.cfi_startproc
	.cfi_offset %r0,0
	.cfi_offset %r1,8
	.cfi_offset %r2,16
	.cfi_offset %r3,24
	.cfi_offset %r4,32
	.cfi_offset %r5,40
	.cfi_offset %r6,48
	.cfi_offset %r7,56
	.cfi_offset %r8,64
	.cfi_offset %r9,72
	.cfi_offset %r10,80
	.cfi_offset %r11,88
	.cfi_offset %r12,96
	.cfi_offset %r13,104
	.cfi_offset %r14,112
	.cfi_offset %r15,120
	.cfi_offset %f0,128
	.cfi_offset %f1,136
	.cfi_offset %f2,144
	.cfi_offset %f3,152
	.cfi_offset %f4,160
	.cfi_offset %f5,168
	.cfi_offset %f6,176
	.cfi_offset %f7,184
	.cfi_offset %f8,192
	.cfi_offset %f9,200
	.cfi_offset %f10,208
	.cfi_offset %f11,216
	.cfi_offset %f12,224
	.cfi_offset %f13,232
	.cfi_offset %f14,240
	.cfi_offset %f15,248
	.cfi_offset %a0,256
	.cfi_offset %a1,260
	.cfi_offset %a2,264
	.cfi_offset %a3,268
	.cfi_offset %a4,272
	.cfi_offset %a5,276
	.cfi_offset %a6,280
	.cfi_offset %a7,284
	.cfi_offset %a8,288
	.cfi_offset %r9,292
	.cfi_offset %a10,296
	.cfi_offset %a11,300
	.cfi_offset %a12,304
	.cfi_offset %a13,308
	.cfi_offset %a14,312
	.cfi_offset %a15,316
	.cfi_offset %c0,318
	.cfi_offset %c1,326
	.cfi_offset %c2,334
	.cfi_offset %c3,342
	.cfi_offset %c4,350
	.cfi_offset %c5,358
	.cfi_offset %c6,366
	.cfi_offset %c7,374
	.cfi_offset %c8,382
	.cfi_offset %c9,390
	.cfi_offset %c10,398
	.cfi_offset %c11,406
	.cfi_offset %c12,414
	.cfi_offset %c13,422
	.cfi_offset %c14,430
	.cfi_offset %c15,438
	.cfi_endproc
