@---
@ Run these test in both Thumb1 and Thumb2 modes, as all of the encodings
@ should be valid, and parse the same, in both.
@---
@ RUN: llvm-mc -triple=thumbv6-apple-darwin -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv7-apple-darwin -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbebv7-unknown-unknown -show-encoding < %s | FileCheck --check-prefix=CHECK-BE %s
  .syntax unified
  .globl _func

@ Check that the assembler can handle the documented syntax from the ARM ARM.
@ For complex constructs like shifter operands, check more thoroughly for them
@ once then spot check that following instructions accept the form generally.
@ This gives us good coverage while keeping the overall size of the test
@ more reasonable.


@ FIXME: Some 3-operand instructions have a 2-operand assembly syntax.

_func:
@ CHECK: _func

@------------------------------------------------------------------------------
@ ADC (register)
@------------------------------------------------------------------------------
        adcs r4, r6

@ CHECK: adcs	r4, r6                  @ encoding: [0x74,0x41]


@------------------------------------------------------------------------------
@ ADD (immediate)
@------------------------------------------------------------------------------
        adds r1, r2, #3
@ When Rd is not explicitly specified, encoding T2 is preferred even though
@ the literal is in the range [0,7] which would allow encoding T1.
        adds r2, #3
        adds r2, #8

@ CHECK: adds	r1, r2, #3              @ encoding: [0xd1,0x1c]
@ CHECK: adds	r2, #3                  @ encoding: [0x03,0x32]
@ CHECK: adds	r2, #8                  @ encoding: [0x08,0x32]


@------------------------------------------------------------------------------
@ ADD (register)
@------------------------------------------------------------------------------
        adds r1, r2, r3
        add r2, r8

@ CHECK: adds	r1, r2, r3              @ encoding: [0xd1,0x18]
@ CHECK: add	r2, r8                  @ encoding: [0x42,0x44]


@------------------------------------------------------------------------------
@ ADD (SP plus immediate)
@------------------------------------------------------------------------------
        add sp, #4
        add sp, #508
        add sp, sp, #4
        add r2, sp, #8
        add r2, sp, #1020
	add sp, sp, #-8
	add sp, #-8

@ CHECK: add	sp, #4                  @ encoding: [0x01,0xb0]
@ CHECK: add	sp, #508                @ encoding: [0x7f,0xb0]
@ CHECK: add	sp, #4                  @ encoding: [0x01,0xb0]
@ CHECK: add	r2, sp, #8              @ encoding: [0x02,0xaa]
@ CHECK: add	r2, sp, #1020           @ encoding: [0xff,0xaa]
@ CHECK: sub	sp, #8                  @ encoding: [0x82,0xb0]
@ CHECK: sub	sp, #8                  @ encoding: [0x82,0xb0]


@------------------------------------------------------------------------------
@ ADD (SP plus register)
@------------------------------------------------------------------------------
        add sp, r3
        add r2, sp, r2

@ CHECK: add	sp, r3                  @ encoding: [0x9d,0x44]
@ CHECK: add	r2, sp, r2              @ encoding: [0x6a,0x44]


@------------------------------------------------------------------------------
@ ADR
@------------------------------------------------------------------------------
        adr r2, _baz
        adr r5, #0
        adr r2, #4
        adr r3, #1020

@ CHECK: adr	r2, _baz                @ encoding: [A,0xa2]
@ CHECK:    @   fixup A - offset: 0, value: _baz, kind: fixup_thumb_adr_pcrel_10
@ CHECK-BE: adr	r2, _baz                @ encoding: [0xa2,A]
@ CHECK-BE:    @   fixup A - offset: 0, value: _baz, kind: fixup_thumb_adr_pcrel_10
@ CHECK: adr	r5, #0                  @ encoding: [0x00,0xa5]
@ CHECK: adr	r2, #4                  @ encoding: [0x01,0xa2]
@ CHECK: adr	r3, #1020               @ encoding: [0xff,0xa3]

@------------------------------------------------------------------------------
@ ASR (immediate)
@------------------------------------------------------------------------------
        asrs r2, r3, #32
        asrs r2, r3, #5
        asrs r2, r3, #1
        asrs r5, #21
        asrs r5, r5, #21
        asrs r3, r5, #21

@ CHECK: asrs	r2, r3, #32             @ encoding: [0x1a,0x10]
@ CHECK: asrs	r2, r3, #5              @ encoding: [0x5a,0x11]
@ CHECK: asrs	r2, r3, #1              @ encoding: [0x5a,0x10]
@ CHECK: asrs	r5, r5, #21             @ encoding: [0x6d,0x15]
@ CHECK: asrs	r5, r5, #21             @ encoding: [0x6d,0x15]
@ CHECK: asrs	r3, r5, #21             @ encoding: [0x6b,0x15]


@------------------------------------------------------------------------------
@ ASR (register)
@------------------------------------------------------------------------------
        asrs r5, r2

@ CHECK: asrs	r5, r2                  @ encoding: [0x15,0x41]


@------------------------------------------------------------------------------
@ B
@------------------------------------------------------------------------------
        b _baz
        beq _bar
        b       #1838
        b       #-420
        beq     #-256
        beq     #160

@ CHECK: b	_baz                    @ encoding: [A,0xe0'A']
@ CHECK:     @   fixup A - offset: 0, value: _baz, kind: fixup_arm_thumb_br
@ CHECK-BE: b	_baz                    @ encoding: [0xe0'A',A]
@ CHECK-BE:     @   fixup A - offset: 0, value: _baz, kind: fixup_arm_thumb_br
@ CHECK: beq	_bar                    @ encoding: [A,0xd0]
@ CHECK:     @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_bcc
@ CHECK-BE: beq	_bar                    @ encoding: [0xd0,A]
@ CHECK-BE:     @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_bcc
@ CHECK: b       #1838                   @ encoding: [0x97,0xe3]
@ CHECK: b       #-420                   @ encoding: [0x2e,0xe7]
@ CHECK: beq     #-256                   @ encoding: [0x80,0xd0]
@ CHECK: beq     #160                    @ encoding: [0x50,0xd0]

@------------------------------------------------------------------------------
@ BL/BLX
@------------------------------------------------------------------------------
        blx     #884800
        blx     #1769600

@ CHECK: blx     #884800                 @ encoding: [0xd8,0xf0,0x20,0xe8]
@ CHECK: blx     #1769600                @ encoding: [0xb0,0xf1,0x40,0xe8]

@------------------------------------------------------------------------------
@ BICS
@------------------------------------------------------------------------------
        bics r1, r6

@ CHECK: bics	r1, r6                  @ encoding: [0xb1,0x43]


@------------------------------------------------------------------------------
@ BKPT
@------------------------------------------------------------------------------
        bkpt #0
        bkpt #255

@ CHECK: bkpt	#0                      @ encoding: [0x00,0xbe]
@ CHECK: bkpt	#255                    @ encoding: [0xff,0xbe]


@------------------------------------------------------------------------------
@ BL/BLX (immediate)
@------------------------------------------------------------------------------
        bl _bar
        blx _baz

@ CHECK: bl	_bar                    @ encoding: [A,0xf0'A',A,0xd0'A']
@ CHECK:     @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_bl
@ CHECK-BE: bl	_bar                    @ encoding: [0xf0'A',A,0xd0'A',A]
@ CHECK-BE:     @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_bl
@ CHECK: blx	_baz                    @ encoding: [A,0xf0'A',A,0xc0'A']
@ CHECK:     @   fixup A - offset: 0, value: _baz, kind: fixup_arm_thumb_blx
@ CHECK-BE: blx	_baz                    @ encoding: [0xf0'A',A,0xc0'A',A]
@ CHECK-BE:     @   fixup A - offset: 0, value: _baz, kind: fixup_arm_thumb_blx


@------------------------------------------------------------------------------
@ BLX (register)
@------------------------------------------------------------------------------
        blx r4

@ CHECK: blx	r4                      @ encoding: [0xa0,0x47]


@------------------------------------------------------------------------------
@ BX
@------------------------------------------------------------------------------
        bx r2

@ CHECK: bx	r2                      @ encoding: [0x10,0x47]


@------------------------------------------------------------------------------
@ CMN
@------------------------------------------------------------------------------

        cmn r5, r1

@ CHECK: cmn	r5, r1                  @ encoding: [0xcd,0x42]


@------------------------------------------------------------------------------
@ CMP
@------------------------------------------------------------------------------
        cmp r6, #32
        cmp r3, r4
        cmp r8, r1

@ CHECK: cmp	r6, #32                 @ encoding: [0x20,0x2e]
@ CHECK: cmp	r3, r4                  @ encoding: [0xa3,0x42]
@ CHECK: cmp	r8, r1                  @ encoding: [0x88,0x45]

@------------------------------------------------------------------------------
@ CPS
@------------------------------------------------------------------------------

        cpsie f
        cpsid a

@ CHECK: cpsie f                        @ encoding: [0x61,0xb6]
@ CHECK: cpsid a                        @ encoding: [0x74,0xb6]

@------------------------------------------------------------------------------
@ EOR
@------------------------------------------------------------------------------
        eors r4, r5

@ CHECK: eors	r4, r5                  @ encoding: [0x6c,0x40]


@------------------------------------------------------------------------------
@ LDM
@------------------------------------------------------------------------------
        ldm r3, {r0, r1, r2, r3, r4, r5, r6, r7}
        ldm r2!, {r1, r3, r4, r5, r7}
        ldm r1, {r1}

@ CHECK: ldm	r3, {r0, r1, r2, r3, r4, r5, r6, r7} @ encoding: [0xff,0xcb]
@ CHECK: ldm	r2!, {r1, r3, r4, r5, r7} @ encoding: [0xba,0xca]
@ CHECK: ldm	r1, {r1}                @ encoding: [0x02,0xc9]


@------------------------------------------------------------------------------
@ LDR (immediate)
@------------------------------------------------------------------------------
        ldr r1, [r5]
        ldr r2, [r6, #32]
        ldr r3, [r7, #124]
        ldr r1, [sp]
        ldr r2, [sp, #24]
        ldr r3, [sp, #1020]


@ CHECK: ldr	r1, [r5]                @ encoding: [0x29,0x68]
@ CHECK: ldr	r2, [r6, #32]           @ encoding: [0x32,0x6a]
@ CHECK: ldr	r3, [r7, #124]          @ encoding: [0xfb,0x6f]
@ CHECK: ldr	r1, [sp]                @ encoding: [0x00,0x99]
@ CHECK: ldr	r2, [sp, #24]           @ encoding: [0x06,0x9a]
@ CHECK: ldr	r3, [sp, #1020]         @ encoding: [0xff,0x9b]


@------------------------------------------------------------------------------
@ LDR (literal)
@------------------------------------------------------------------------------
        ldr r1, _foo
        ldr     r3, #604
        ldr     r3, #368

@ CHECK: ldr	r1, _foo                @ encoding: [A,0x49]
@ CHECK:     @   fixup A - offset: 0, value: _foo, kind: fixup_arm_thumb_cp
@ CHECK-BE: ldr	r1, _foo                @ encoding: [0x49,A]
@ CHECK-BE:     @   fixup A - offset: 0, value: _foo, kind: fixup_arm_thumb_cp
@ CHECK: ldr     r3, [pc, #604]         @ encoding: [0x97,0x4b]
@ CHECK: ldr     r3, [pc, #368]         @ encoding: [0x5c,0x4b]

@------------------------------------------------------------------------------
@ LDR (register)
@------------------------------------------------------------------------------
        ldr r1, [r2, r3]

@ CHECK: ldr	r1, [r2, r3]            @ encoding: [0xd1,0x58]


@------------------------------------------------------------------------------
@ LDRB (immediate)
@------------------------------------------------------------------------------
        ldrb r4, [r3]
        ldrb r5, [r6, #0]
        ldrb r6, [r7, #31]

@ CHECK: ldrb	r4, [r3]                @ encoding: [0x1c,0x78]
@ CHECK: ldrb	r5, [r6]                @ encoding: [0x35,0x78]
@ CHECK: ldrb	r6, [r7, #31]           @ encoding: [0xfe,0x7f]


@------------------------------------------------------------------------------
@ LDRB (register)
@------------------------------------------------------------------------------
        ldrb r6, [r4, r5]

@ CHECK: ldrb	r6, [r4, r5]            @ encoding: [0x66,0x5d]


@------------------------------------------------------------------------------
@ LDRH (immediate)
@------------------------------------------------------------------------------
        ldrh r3, [r3]
        ldrh r4, [r6, #2]
        ldrh r5, [r7, #62]

@ CHECK: ldrh	r3, [r3]                @ encoding: [0x1b,0x88]
@ CHECK: ldrh	r4, [r6, #2]            @ encoding: [0x74,0x88]
@ CHECK: ldrh	r5, [r7, #62]           @ encoding: [0xfd,0x8f]


@------------------------------------------------------------------------------
@ LDRH (register)
@------------------------------------------------------------------------------
        ldrh r6, [r2, r6]

@ CHECK: ldrh	r6, [r2, r6]            @ encoding: [0x96,0x5b]


@------------------------------------------------------------------------------
@ LDRSB/LDRSH
@------------------------------------------------------------------------------
        ldrsb r6, [r2, r6]
        ldrsh r3, [r7, r1]

@ CHECK: ldrsb	r6, [r2, r6]            @ encoding: [0x96,0x57]
@ CHECK: ldrsh	r3, [r7, r1]            @ encoding: [0x7b,0x5e]


@------------------------------------------------------------------------------
@ LSL (immediate)
@------------------------------------------------------------------------------
        lsls r4, r5, #0
        lsls r4, r5, #4
        lsls r3, #12
        lsls r3, r3, #12
        lsls r1, r3, #12

@ CHECK: lsls	r4, r5, #0              @ encoding: [0x2c,0x00]
@ CHECK: lsls	r4, r5, #4              @ encoding: [0x2c,0x01]
@ CHECK: lsls	r3, r3, #12             @ encoding: [0x1b,0x03]
@ CHECK: lsls	r3, r3, #12             @ encoding: [0x1b,0x03]
@ CHECK: lsls	r1, r3, #12             @ encoding: [0x19,0x03]


@------------------------------------------------------------------------------
@ LSL (register)
@------------------------------------------------------------------------------
        lsls r2, r6

@ CHECK: lsls	r2, r6                  @ encoding: [0xb2,0x40]


@------------------------------------------------------------------------------
@ LSR (immediate)
@------------------------------------------------------------------------------
        lsrs r1, r3, #1
        lsrs r1, r3, #32
        lsrs r4, #20
        lsrs r4, r4, #20
        lsrs r2, r4, #20

@ CHECK: lsrs	r1, r3, #1              @ encoding: [0x59,0x08]
@ CHECK: lsrs	r1, r3, #32             @ encoding: [0x19,0x08]
@ CHECK: lsrs	r4, r4, #20             @ encoding: [0x24,0x0d]
@ CHECK: lsrs	r4, r4, #20             @ encoding: [0x24,0x0d]
@ CHECK: lsrs	r2, r4, #20             @ encoding: [0x22,0x0d]


@------------------------------------------------------------------------------
@ LSR (register)
@------------------------------------------------------------------------------
        lsrs r2, r6

@ CHECK: lsrs	r2, r6                  @ encoding: [0xf2,0x40]


@------------------------------------------------------------------------------
@ MOV (immediate)
@------------------------------------------------------------------------------
        movs r2, #0
        movs r2, #255
        movs r2, #23

@ CHECK: movs	r2, #0                  @ encoding: [0x00,0x22]
@ CHECK: movs	r2, #255                @ encoding: [0xff,0x22]
@ CHECK: movs	r2, #23                 @ encoding: [0x17,0x22]


@------------------------------------------------------------------------------
@ MOV (register)
@------------------------------------------------------------------------------
        mov r3, r4
        movs r1, r3

@ CHECK: mov	r3, r4                  @ encoding: [0x23,0x46]
@ CHECK: movs	r1, r3                  @ encoding: [0x19,0x00]


@------------------------------------------------------------------------------
@ MUL
@------------------------------------------------------------------------------
        muls r1, r2, r1
        muls r2, r2, r3
        muls r3, r4

@ CHECK: muls	r1, r2, r1              @ encoding: [0x51,0x43]
@ CHECK: muls	r2, r3, r2              @ encoding: [0x5a,0x43]
@ CHECK: muls	r3, r4, r3              @ encoding: [0x63,0x43]


@------------------------------------------------------------------------------
@ MVN
@------------------------------------------------------------------------------
        mvns r6, r3

@ CHECK: mvns	r6, r3                  @ encoding: [0xde,0x43]


@------------------------------------------------------------------------------
@ NEG
@------------------------------------------------------------------------------
        negs r3, r4

@ CHECK: rsbs	r3, r4, #0              @ encoding: [0x63,0x42]

@------------------------------------------------------------------------------
@ ORR
@------------------------------------------------------------------------------
        orrs  r3, r4

@ CHECK-ERRORS: 	orrs	r3, r4                  @ encoding: [0x23,0x43]


@------------------------------------------------------------------------------
@ POP
@------------------------------------------------------------------------------
        pop {r2, r3, r6}

@ CHECK: pop	{r2, r3, r6}            @ encoding: [0x4c,0xbc]


@------------------------------------------------------------------------------
@ PUSH
@------------------------------------------------------------------------------
        push {r1, r2, r7}

@ CHECK: push	{r1, r2, r7}            @ encoding: [0x86,0xb4]


@------------------------------------------------------------------------------
@ REV/REV16/REVSH
@------------------------------------------------------------------------------
        rev r6, r3
        rev16 r7, r2
        revsh r5, r1

@ CHECK: rev	r6, r3                  @ encoding: [0x1e,0xba]
@ CHECK: rev16	r7, r2                  @ encoding: [0x57,0xba]
@ CHECK: revsh	r5, r1                  @ encoding: [0xcd,0xba]


@------------------------------------------------------------------------------
@ ROR
@------------------------------------------------------------------------------
        rors r2, r7

@ CHECK: rors	r2, r7                  @ encoding: [0xfa,0x41]


@------------------------------------------------------------------------------
@ RSB
@------------------------------------------------------------------------------
        rsbs r1, r3, #0

@ CHECK: rsbs	r1, r3, #0              @ encoding: [0x59,0x42]


@------------------------------------------------------------------------------
@ SBC
@------------------------------------------------------------------------------
        sbcs r4, r3

@ CHECK: sbcs	r4, r3                  @ encoding: [0x9c,0x41]


@------------------------------------------------------------------------------
@ SETEND
@------------------------------------------------------------------------------
        setend be
        setend le

@ CHECK: setend	be                      @ encoding: [0x58,0xb6]
@ CHECK: setend	le                      @ encoding: [0x50,0xb6]


@------------------------------------------------------------------------------
@ STM
@------------------------------------------------------------------------------
        stm r1!, {r2, r6}
        stm r1!, {r1, r2, r3, r7}

@ CHECK: stm	r1!, {r2, r6}           @ encoding: [0x44,0xc1]
@ CHECK: stm	r1!, {r1, r2, r3, r7}   @ encoding: [0x8e,0xc1]


@------------------------------------------------------------------------------
@ STR (immediate)
@------------------------------------------------------------------------------
        str r2, [r7]
        str r2, [r7, #0]
        str r5, [r1, #4]
        str r3, [r7, #124]
        str r2, [sp]
        str r3, [sp, #0]
        str r4, [sp, #20]
        str r5, [sp, #1020]

@ CHECK: str	r2, [r7]                @ encoding: [0x3a,0x60]
@ CHECK: str	r2, [r7]                @ encoding: [0x3a,0x60]
@ CHECK: str	r5, [r1, #4]            @ encoding: [0x4d,0x60]
@ CHECK: str	r3, [r7, #124]          @ encoding: [0xfb,0x67]
@ CHECK: str	r2, [sp]                @ encoding: [0x00,0x92]
@ CHECK: str	r3, [sp]                @ encoding: [0x00,0x93]
@ CHECK: str	r4, [sp, #20]           @ encoding: [0x05,0x94]
@ CHECK: str	r5, [sp, #1020]         @ encoding: [0xff,0x95]


@------------------------------------------------------------------------------
@ STR (register)
@------------------------------------------------------------------------------
        str r2, [r7, r3]

@ CHECK: str	r2, [r7, r3]            @ encoding: [0xfa,0x50]


@------------------------------------------------------------------------------
@ STRB (immediate)
@------------------------------------------------------------------------------
        strb r4, [r3]
        strb r5, [r6, #0]
        strb r6, [r7, #31]

@ CHECK: strb	r4, [r3]                @ encoding: [0x1c,0x70]
@ CHECK: strb	r5, [r6]                @ encoding: [0x35,0x70]
@ CHECK: strb	r6, [r7, #31]           @ encoding: [0xfe,0x77]


@------------------------------------------------------------------------------
@ STRB (register)
@------------------------------------------------------------------------------
        strb r6, [r4, r5]

@ CHECK: strb	r6, [r4, r5]            @ encoding: [0x66,0x55]


@------------------------------------------------------------------------------
@ STRH (immediate)
@------------------------------------------------------------------------------
        strh r3, [r3]
        strh r4, [r6, #2]
        strh r5, [r7, #62]

@ CHECK: strh	r3, [r3]                @ encoding: [0x1b,0x80]
@ CHECK: strh	r4, [r6, #2]            @ encoding: [0x74,0x80]
@ CHECK: strh	r5, [r7, #62]           @ encoding: [0xfd,0x87]


@------------------------------------------------------------------------------
@ STRH (register)
@------------------------------------------------------------------------------
        strh r6, [r2, r6]

@ CHECK: strh	r6, [r2, r6]            @ encoding: [0x96,0x53]


@------------------------------------------------------------------------------
@ SUB (immediate)
@------------------------------------------------------------------------------
        subs r1, r2, #3
        subs r2, #3
        subs r2, #8

@ CHECK: subs	r1, r2, #3              @ encoding: [0xd1,0x1e]
@ CHECK: subs	r2, #3                  @ encoding: [0x03,0x3a]
@ CHECK: subs	r2, #8                  @ encoding: [0x08,0x3a]


@------------------------------------------------------------------------------
@ SUB (SP minus immediate)
@------------------------------------------------------------------------------
        sub sp, #12
        sub sp, sp, #508

@ CHECK: sub	sp, #12                 @ encoding: [0x83,0xb0]
@ CHECK: sub	sp, #508                @ encoding: [0xff,0xb0]


@------------------------------------------------------------------------------
@ SUB (register)
@------------------------------------------------------------------------------
        subs r1, r2, r3

@ CHECK: subs	r1, r2, r3              @ encoding: [0xd1,0x1a]


@------------------------------------------------------------------------------
@ SVC
@------------------------------------------------------------------------------
        svc #0
        svc #255

@ CHECK: svc	#0                      @ encoding: [0x00,0xdf]
@ CHECK: svc	#255                    @ encoding: [0xff,0xdf]


@------------------------------------------------------------------------------
@ SXTB/SXTH
@------------------------------------------------------------------------------
        sxtb r3, r5
        sxth r3, r5

@ CHECK: sxtb	r3, r5                  @ encoding: [0x6b,0xb2]
@ CHECK: sxth	r3, r5                  @ encoding: [0x2b,0xb2]


@------------------------------------------------------------------------------
@ TST
@------------------------------------------------------------------------------
        tst r6, r1

@ CHECK: tst	r6, r1                  @ encoding: [0x0e,0x42]


@------------------------------------------------------------------------------
@ UXTB/UXTH
@------------------------------------------------------------------------------
        uxtb  r7, r2
        uxth  r1, r4

@ CHECK: uxtb	r7, r2                  @ encoding: [0xd7,0xb2]
@ CHECK: uxth	r1, r4                  @ encoding: [0xa1,0xb2]


