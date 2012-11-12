@ RUN: llvm-mc -triple thumbv6-apple-darwin -show-encoding < %s | FileCheck %s
        .code 16

        cmp     r1, r2
@ CHECK: cmp	r1, r2                  @ encoding: [0x91,0x42]

        pop     {r1, r2, r4}
@ CHECK: pop    {r1, r2, r4}            @ encoding: [0x16,0xbc]

        trap
@ CHECK: trap                           @ encoding: [0xfe,0xde]

	blx	r9
        blx r10
@ CHECK: blx	r9                      @ encoding: [0xc8,0x47]
@ CHECK: blx	r10                     @ encoding: [0xd0,0x47]

        rev     r2, r3
        rev16   r3, r4
        revsh   r5, r6
@ CHECK: rev	r2, r3                  @ encoding: [0x1a,0xba]
@ CHECK: rev16	r3, r4                  @ encoding: [0x63,0xba]
@ CHECK: revsh	r5, r6                  @ encoding: [0xf5,0xba]

	sxtb	r2, r3
	sxth	r2, r3
@ CHECK: sxtb	r2, r3                  @ encoding: [0x5a,0xb2]
@ CHECK: sxth	r2, r3                  @ encoding: [0x1a,0xb2]

	tst	r4, r5
@ CHECK: tst	r4, r5                  @ encoding: [0x2c,0x42]

	uxtb	r3, r6
	uxth	r3, r6
@ CHECK: uxtb	r3, r6                  @ encoding: [0xf3,0xb2]
@ CHECK: uxth	r3, r6                  @ encoding: [0xb3,0xb2]

	ldr	r3, [r1, r2]
@ CHECK: ldr	r3, [r1, r2]            @ encoding: [0x8b,0x58]

        bkpt  #2
@ CHECK: bkpt  #2                       @ encoding: [0x02,0xbe]

        nop
@ CHECK: nop @ encoding: [0xc0,0x46]

        cpsie aif
@ CHECK: cpsie aif                      @ encoding: [0x67,0xb6]

        mov  r0, pc
@ CHECK: mov  r0, pc                    @ encoding: [0x78,0x46]
