@ RUN: llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbebv7-unknown-unknown -mcpu=cortex-a8 -show-encoding < %s | FileCheck --check-prefix=CHECK-BE %s
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
@ ADC (immediate)
@------------------------------------------------------------------------------
        adc r0, r1, #4
        adcs r0, r1, #0
        adc r1, r2, #255
        adc r3, r7, #0x00550055
        adc r8, r12, #0xaa00aa00
        adc r9, r7, #0xa5a5a5a5
        adc r5, r3, #0x87000000
        adc r4, r2, #0x7f800000
        adc r4, r2, #0x00000680

@ CHECK: adc	r0, r1, #4              @ encoding: [0x41,0xf1,0x04,0x00]
@ CHECK: adcs	r0, r1, #0              @ encoding: [0x51,0xf1,0x00,0x00]
@ CHECK: adc	r1, r2, #255            @ encoding: [0x42,0xf1,0xff,0x01]
@ CHECK: adc	r3, r7, #5570645        @ encoding: [0x47,0xf1,0x55,0x13]
@ CHECK: adc	r8, r12, #2852170240    @ encoding: [0x4c,0xf1,0xaa,0x28]
@ CHECK: adc	r9, r7, #2779096485     @ encoding: [0x47,0xf1,0xa5,0x39]
@ CHECK: adc	r5, r3, #2264924160     @ encoding: [0x43,0xf1,0x07,0x45]
@ CHECK: adc	r4, r2, #2139095040     @ encoding: [0x42,0xf1,0xff,0x44]
@ CHECK: adc	r4, r2, #1664           @ encoding: [0x42,0xf5,0xd0,0x64]

@------------------------------------------------------------------------------
@ ADC (register)
@------------------------------------------------------------------------------
        adc r4, r5, r6
        adcs r4, r5, r6
        adc.w r9, r1, r3
        adcs.w r9, r1, r3
        adc	r0, r1, r3, ror #4
        adcs	r0, r1, r3, lsl #7
        adc.w	r0, r1, r3, lsr #31
        adcs.w	r0, r1, r3, asr #32

@ CHECK: adc.w	r4, r5, r6              @ encoding: [0x45,0xeb,0x06,0x04]
@ CHECK: adcs.w	r4, r5, r6              @ encoding: [0x55,0xeb,0x06,0x04]
@ CHECK: adc.w	r9, r1, r3              @ encoding: [0x41,0xeb,0x03,0x09]
@ CHECK: adcs.w	r9, r1, r3              @ encoding: [0x51,0xeb,0x03,0x09]
@ CHECK: adc.w	r0, r1, r3, ror #4      @ encoding: [0x41,0xeb,0x33,0x10]
@ CHECK: adcs.w	r0, r1, r3, lsl #7      @ encoding: [0x51,0xeb,0xc3,0x10]
@ CHECK: adc.w	r0, r1, r3, lsr #31     @ encoding: [0x41,0xeb,0xd3,0x70]
@ CHECK: adcs.w	r0, r1, r3, asr #32     @ encoding: [0x51,0xeb,0x23,0x00]


@------------------------------------------------------------------------------
@ ADD (immediate)
@------------------------------------------------------------------------------
        itet eq
        addeq r1, r2, #4
        addwne r5, r3, #1023
        addeq r4, r5, #293
        add r2, sp, #1024
        add r2, r8, #0xff00
        add r2, r3, #257
        addw r2, r3, #257
        add r12, r6, #0x100
        addw r12, r6, #0x100
        adds r1, r2, #0x1f0
	add r2, #1
        add r0, r0, #32
        adds r2, r2, #56
        adds r2, #56
        add r1, r7, #0xcbcbcbcb
        add sp, sp, #0x1fe0000

        adds.w r2, #-16
        adds.w r2, r2, #-16
        addw r2, #-16
        addw r2, #-16
        addw r2, r2, #-16

@ CHECK: itet	eq                      @ encoding: [0x0a,0xbf]
@ CHECK: addeq	r1, r2, #4              @ encoding: [0x11,0x1d]
@ CHECK: addwne	r5, r3, #1023           @ encoding: [0x03,0xf2,0xff,0x35]
@ CHECK: addweq	r4, r5, #293            @ encoding: [0x05,0xf2,0x25,0x14]
@ CHECK: add.w	r2, sp, #1024           @ encoding: [0x0d,0xf5,0x80,0x62]
@ CHECK: add.w	r2, r8, #65280          @ encoding: [0x08,0xf5,0x7f,0x42]
@ CHECK: addw	r2, r3, #257            @ encoding: [0x03,0xf2,0x01,0x12]
@ CHECK: addw	r2, r3, #257            @ encoding: [0x03,0xf2,0x01,0x12]
@ CHECK: add.w	r12, r6, #256           @ encoding: [0x06,0xf5,0x80,0x7c]
@ CHECK: addw	r12, r6, #256           @ encoding: [0x06,0xf2,0x00,0x1c]
@ CHECK: adds.w	r1, r2, #496            @ encoding: [0x12,0xf5,0xf8,0x71]
@ CHECK: add.w	r2, r2, #1              @ encoding: [0x02,0xf1,0x01,0x02]
@ CHECK: add.w	r0, r0, #32             @ encoding: [0x00,0xf1,0x20,0x00]
@ CHECK: adds	r2, #56                 @ encoding: [0x38,0x32]
@ CHECK: adds	r2, #56                 @ encoding: [0x38,0x32]
@ CHECK: add.w  r1, r7, #3419130827     @ encoding: [0x07,0xf1,0xcb,0x31]
@ CHECK: add.w	sp, sp, #33423360       @ encoding: [0x0d,0xf1,0xff,0x7d]

@ CHECK: subs.w	r2, r2, #16             @ encoding: [0xb2,0xf1,0x10,0x02]
@ CHECK: subs.w	r2, r2, #16             @ encoding: [0xb2,0xf1,0x10,0x02]
@ CHECK: subw	r2, r2, #16             @ encoding: [0xa2,0xf2,0x10,0x02]
@ CHECK: subw	r2, r2, #16             @ encoding: [0xa2,0xf2,0x10,0x02]
@ CHECK: subw	r2, r2, #16             @ encoding: [0xa2,0xf2,0x10,0x02]


@------------------------------------------------------------------------------
@ ADD (register, not SP) A8.8.6
@------------------------------------------------------------------------------
        add r1, r2, r8
        add r5, r9, r2, asr #32
        adds r7, r3, r1, lsl #31
        adds.w r0, r3, r6, lsr #25
        add.w r4, r8, r1, ror #12
        adds r1, r1, r7              // T1
        it eq
        addeq r1, r3, r5             // T1
        it eq
        addeq r1, r1, r5             // T1
        it eq
        addseq r1, r3, r5            // T3
        it eq
        addseq r1, r1, r5            // T3
        add r10, r8
        add r10, r10, r8
        it eq
        addeq r1, r10                // T2
        it eq
        addseq r1, r10               // T3

@ CHECK: add.w	r1, r2, r8              @ encoding: [0x02,0xeb,0x08,0x01]
@ CHECK: add.w	r5, r9, r2, asr #32     @ encoding: [0x09,0xeb,0x22,0x05]
@ CHECK: adds.w	r7, r3, r1, lsl #31     @ encoding: [0x13,0xeb,0xc1,0x77]
@ CHECK: adds.w	r0, r3, r6, lsr #25     @ encoding: [0x13,0xeb,0x56,0x60]
@ CHECK: add.w	r4, r8, r1, ror #12     @ encoding: [0x08,0xeb,0x31,0x34]
@ CHECK: adds r1, r1, r7                @ encoding: [0xc9,0x19]
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
@ CHECK: addeq r1, r3, r5               @ encoding: [0x59,0x19]
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
@ CHECK: addeq r1, r1, r5               @ encoding: [0x49,0x19]
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
@ CHECK: addseq.w r1, r3, r5            @ encoding: [0x13,0xeb,0x05,0x01]
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
@ CHECK: addseq.w r1, r1, r5            @ encoding: [0x11,0xeb,0x05,0x01]
@ CHECK: add	r10, r8                 @ encoding: [0xc2,0x44]
@ CHECK: add	r10, r8                 @ encoding: [0xc2,0x44]
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
@ CHECK: addeq r1, r10                  @ encoding: [0x51,0x44]
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
@ CHECK: addseq.w r1, r1, r10           @ encoding: [0x11,0xeb,0x0a,0x01]

@------------------------------------------------------------------------------
@ ADD (SP plus immediate) A8.8.9
@------------------------------------------------------------------------------
        it eq
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
        addeq r7, sp, #1020          // T1
@ CHECK: addeq	r7, sp, #1020           @ encoding: [0xff,0xaf]

        it eq
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
        addeq sp, sp, #508           // T2
@ FIXME: ARMARM says 'addeq sp, sp, #508'
@ CHECK: addeq	sp, #508                @ encoding: [0x7f,0xb0]

        add r7, sp, #15              // T3
@ CHECK: add.w	r7, sp, #15             @ encoding: [0x0d,0xf1,0x0f,0x07]
        adds r7, sp, #16             // T3
@ CHECK: adds.w	r7, sp, #16             @ encoding: [0x1d,0xf1,0x10,0x07]
        add r8, sp, #16              // T3
@ CHECK: add.w	r8, sp, #16             @ encoding: [0x0d,0xf1,0x10,0x08]

        addw r6, sp, #1020           // T4
@ CHECK: addw	r6, sp, #1020           @ encoding: [0x0d,0xf2,0xfc,0x36]
        add r6, sp, #1019            // T4
@ CHECK: addw	r6, sp, #1019           @ encoding: [0x0d,0xf2,0xfb,0x36]

@------------------------------------------------------------------------------
@ ADD (SP plus register) A8.8.10
@------------------------------------------------------------------------------
        it eq
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
        addeq r8, sp, r8             // T1
@ CHECK: addeq	r8, sp, r8              @ encoding: [0xe8,0x44]
        it eq
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
        addeq r8, sp                 // T1
@ CHECK: addeq	r8, sp                  @ encoding: [0xe8,0x44]

        it eq
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
        addeq sp, r9                 // T2
@ CHECK: addeq	sp, r9                  @ encoding: [0xcd,0x44]

        add r2, sp, ip               // T3
@ CHECK: add.w r2, sp, r12              @ encoding: [0x0d,0xeb,0x0c,0x02]
        it eq
@ CHECK: it eq                          @ encoding: [0x08,0xbf]
        addeq r2, sp, ip             // T3
@ CHECK: addeq.w r2, sp, r12            @ encoding: [0x0d,0xeb,0x0c,0x02]


@------------------------------------------------------------------------------
@ FIXME: ADR
@------------------------------------------------------------------------------

        subw r11, pc, #3270
        adr.w r2, #3
        adr.w r11, #-826
        adr.w r1, #-0x0

@ CHECK: subw  r11, pc, #3270          @ encoding: [0xaf,0xf6,0xc6,0x4b]
@ CHECK: adr.w r2, #3                  @ encoding: [0x0f,0xf2,0x03,0x02]
@ CHECK: adr.w r11, #-826              @ encoding: [0xaf,0xf2,0x3a,0x3b]
@ CHECK: adr.w r1, #-0                 @ encoding: [0xaf,0xf2,0x00,0x01]

@------------------------------------------------------------------------------
@ AND (immediate)
@------------------------------------------------------------------------------
        and r2, r5, #0xff000
        ands r3, r12, #0xf
        and r1, #0xff
        and r1, r1, #0xff
        and r5, r4, #0xffffffff
        ands r1, r9, #0xffffffff

@ CHECK: and	r2, r5, #1044480        @ encoding: [0x05,0xf4,0x7f,0x22]
@ CHECK: ands	r3, r12, #15            @ encoding: [0x1c,0xf0,0x0f,0x03]
@ CHECK: and	r1, r1, #255            @ encoding: [0x01,0xf0,0xff,0x01]
@ CHECK: and	r1, r1, #255            @ encoding: [0x01,0xf0,0xff,0x01]
@ CHECK: and	r5, r4, #4294967295     @ encoding: [0x04,0xf0,0xff,0x35]
@ CHECK: ands	r1, r9, #4294967295     @ encoding: [0x19,0xf0,0xff,0x31]

@------------------------------------------------------------------------------
@ AND (register)
@------------------------------------------------------------------------------
        and r4, r9, r8
        and r1, r4, r8, asr #3
        ands r2, r1, r7, lsl #1
        ands.w r4, r5, r2, lsr #20
        and.w r9, r12, r1, ror #17

@ CHECK: and.w	r4, r9, r8              @ encoding: [0x09,0xea,0x08,0x04]
@ CHECK: and.w	r1, r4, r8, asr #3      @ encoding: [0x04,0xea,0xe8,0x01]
@ CHECK: ands.w	r2, r1, r7, lsl #1      @ encoding: [0x11,0xea,0x47,0x02]
@ CHECK: ands.w	r4, r5, r2, lsr #20     @ encoding: [0x15,0xea,0x12,0x54]
@ CHECK: and.w	r9, r12, r1, ror #17    @ encoding: [0x0c,0xea,0x71,0x49]

@------------------------------------------------------------------------------
@ ASR (immediate)
@------------------------------------------------------------------------------
        asr r2, r3, #12
        asrs r8, r3, #32
        asrs.w r2, r3, #1
        asr r2, r3, #4
        asrs r2, r12, #15

        asr r3, #19
        asrs r8, #2
        asrs.w r7, #5
        asr.w r12, #21

        asrs  r1, r2, #1
        itt eq
        asrseq r1, r2, #1
        asreq r1, r2, #1

@ CHECK: asr.w	r2, r3, #12             @ encoding: [0x4f,0xea,0x23,0x32]
@ CHECK: asrs.w	r8, r3, #32             @ encoding: [0x5f,0xea,0x23,0x08]
@ CHECK: asrs.w	r2, r3, #1              @ encoding: [0x5f,0xea,0x63,0x02]
@ CHECK: asr.w	r2, r3, #4              @ encoding: [0x4f,0xea,0x23,0x12]
@ CHECK: asrs.w	r2, r12, #15            @ encoding: [0x5f,0xea,0xec,0x32]

@ CHECK: asr.w	r3, r3, #19             @ encoding: [0x4f,0xea,0xe3,0x43]
@ CHECK: asrs.w	r8, r8, #2              @ encoding: [0x5f,0xea,0xa8,0x08]
@ CHECK: asrs.w	r7, r7, #5              @ encoding: [0x5f,0xea,0x67,0x17]
@ CHECK: asr.w	r12, r12, #21           @ encoding: [0x4f,0xea,0x6c,0x5c]

@ CHECK: asrs   r1, r2, #1              @ encoding: [0x51,0x10]
@ CHECK: itt    eq                      @ encoding: [0x04,0xbf]
@ CHECK: asrseq.w r1, r2, #1            @ encoding: [0x5f,0xea,0x62,0x01]
@ CHECK: asreq  r1, r2, #1              @ encoding: [0x51,0x10]

@------------------------------------------------------------------------------
@ ASR (register)
@------------------------------------------------------------------------------
        asr r3, r4, r2
        asr.w r1, r2
        asrs r3, r4, r8

@ CHECK: asr.w	r3, r4, r2              @ encoding: [0x44,0xfa,0x02,0xf3]
@ CHECK: asr.w	r1, r1, r2              @ encoding: [0x41,0xfa,0x02,0xf1]
@ CHECK: asrs.w	r3, r4, r8              @ encoding: [0x54,0xfa,0x08,0xf3]


@------------------------------------------------------------------------------
@ B
@------------------------------------------------------------------------------
        b.w   _bar
        beq.w   _bar
        it eq
        beq.w _bar
        bmi.w   #-183396

@ CHECK: b.w	_bar                    @ encoding: [A,0xf0'A',A,0x90'A']
@ CHECK:  @   fixup A - offset: 0, value: _bar, kind: fixup_t2_uncondbranch
@ CHECK-BE: b.w	_bar                    @ encoding: [0xf0'A',A,0x90'A',A]
@ CHECK-BE:  @   fixup A - offset: 0, value: _bar, kind: fixup_t2_uncondbranch
@ CHECK: beq.w	_bar                    @ encoding: [A,0xf0'A',A,0x80'A']
@ CHECK:  @   fixup A - offset: 0, value: _bar, kind: fixup_t2_condbranch
@ CHECK-BE: beq.w	_bar                    @ encoding: [0xf0'A',A,0x80'A',A]
@ CHECK-BE:  @   fixup A - offset: 0, value: _bar, kind: fixup_t2_condbranch
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: beq.w	_bar                    @ encoding: [A,0xf0'A',A,0x90'A']
@ CHECK:  @   fixup A - offset: 0, value: _bar, kind: fixup_t2_uncondbranch
@ CHECK-BE: beq.w	_bar                    @ encoding: [0xf0'A',A,0x90'A',A]
@ CHECK-BE:  @   fixup A - offset: 0, value: _bar, kind: fixup_t2_uncondbranch
@ CHECK: bmi.w   #-183396                @ encoding: [0x13,0xf5,0xce,0xa9]


@------------------------------------------------------------------------------
@ BFC
@------------------------------------------------------------------------------
        bfc r5, #3, #17
        it lo
        bfccc r5, #3, #17

@ CHECK: bfc	r5, #3, #17             @ encoding: [0x6f,0xf3,0xd3,0x05]
@ CHECK: it	lo                      @ encoding: [0x38,0xbf]
@ CHECK: bfclo	r5, #3, #17             @ encoding: [0x6f,0xf3,0xd3,0x05]


@------------------------------------------------------------------------------
@ BFI
@------------------------------------------------------------------------------
        bfi r5, r2, #3, #17
        it ne
        bfine r5, r2, #3, #17

@ CHECK: bfi	r5, r2, #3, #17         @ encoding: [0x62,0xf3,0xd3,0x05]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: bfine	r5, r2, #3, #17         @ encoding: [0x62,0xf3,0xd3,0x05]


@------------------------------------------------------------------------------
@ BIC
@------------------------------------------------------------------------------
        bic r10, r1, #0xf
        bic r5, r2, #0xffffffff
        bics r11, r10, #0xffffffff
        bic r12, r3, r6
        bic r11, r2, r6, lsl #12
        bic r8, r4, r1, lsr #11
        bic r7, r5, r7, lsr #15
        bic r6, r7, r9, asr #32
        bic r5, r6, r8, ror #1

        @ destination register is optional
        bic r1, #0xf
        bic r1, r1
        bic r4, r2, lsl #31
        bic r6, r3, lsr #12
        bic r7, r4, lsr #7
        bic r8, r5, asr #15
        bic r12, r6, ror #29

@ CHECK: bic	r10, r1, #15            @ encoding: [0x21,0xf0,0x0f,0x0a]
@ CHECK: bic	r5, r2, #4294967295     @ encoding: [0x22,0xf0,0xff,0x35]
@ CHECK: bics	r11, r10, #4294967295   @ encoding: [0x3a,0xf0,0xff,0x3b]
@ CHECK: bic.w	r12, r3, r6             @ encoding: [0x23,0xea,0x06,0x0c]
@ CHECK: bic.w	r11, r2, r6, lsl #12    @ encoding: [0x22,0xea,0x06,0x3b]
@ CHECK: bic.w	r8, r4, r1, lsr #11     @ encoding: [0x24,0xea,0xd1,0x28]
@ CHECK: bic.w	r7, r5, r7, lsr #15     @ encoding: [0x25,0xea,0xd7,0x37]
@ CHECK: bic.w	r6, r7, r9, asr #32     @ encoding: [0x27,0xea,0x29,0x06]
@ CHECK: bic.w	r5, r6, r8, ror #1      @ encoding: [0x26,0xea,0x78,0x05]

@ CHECK: bic	r1, r1, #15             @ encoding: [0x21,0xf0,0x0f,0x01]
@ CHECK: bic.w	r1, r1, r1              @ encoding: [0x21,0xea,0x01,0x01]
@ CHECK: bic.w	r4, r4, r2, lsl #31     @ encoding: [0x24,0xea,0xc2,0x74]
@ CHECK: bic.w	r6, r6, r3, lsr #12     @ encoding: [0x26,0xea,0x13,0x36]
@ CHECK: bic.w	r7, r7, r4, lsr #7      @ encoding: [0x27,0xea,0xd4,0x17]
@ CHECK: bic.w	r8, r8, r5, asr #15     @ encoding: [0x28,0xea,0xe5,0x38]
@ CHECK: bic.w	r12, r12, r6, ror #29   @ encoding: [0x2c,0xea,0x76,0x7c]

@------------------------------------------------------------------------------
@ BKPT
@------------------------------------------------------------------------------
        it pl
        bkpt #234

@ CHECK: it pl                      @ encoding: [0x58,0xbf]
@ CHECK: bkpt #234                    @ encoding: [0xea,0xbe]

@------------------------------------------------------------------------------
@ BXJ
@------------------------------------------------------------------------------
        bxj r5
        it ne
        bxjne r7

@ CHECK: bxj	r5                      @ encoding: [0xc5,0xf3,0x00,0x8f]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: bxjne	r7                      @ encoding: [0xc7,0xf3,0x00,0x8f]


@------------------------------------------------------------------------------
@ CBZ/CBNZ
@------------------------------------------------------------------------------
        cbnz    r7, #6
        cbnz    r7, #12
        cbz   r6, _bar
        cbnz   r6, _bar

@ CHECK: cbnz    r7, #6                  @ encoding: [0x1f,0xb9]
@ CHECK: cbnz    r7, #12                 @ encoding: [0x37,0xb9]
@ CHECK: cbz	r6, _bar                @ encoding: [0x06'A',0xb1'A']
@ CHECK:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_cb
@ CHECK-BE: cbz	r6, _bar                @ encoding: [0xb1'A',0x06'A']
@ CHECK-BE:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_cb
@ CHECK: cbnz	r6, _bar                @ encoding: [0x06'A',0xb9'A']
@ CHECK:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_cb
@ CHECK-BE: cbnz	r6, _bar                @ encoding: [0xb9'A',0x06'A']
@ CHECK-BE:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_thumb_cb


@------------------------------------------------------------------------------
@ CDP/CDP2
@------------------------------------------------------------------------------
  cdp  p7, #1, c1, c1, c1, #4
  cdp2  p7, #1, c1, c1, c1, #4

@ CHECK: cdp	p7, #1, c1, c1, c1, #4  @ encoding: [0x11,0xee,0x81,0x17]
@ CHECK: cdp2	p7, #1, c1, c1, c1, #4  @ encoding: [0x11,0xfe,0x81,0x17]


@------------------------------------------------------------------------------
@ CLREX
@------------------------------------------------------------------------------
        clrex
        it ne
        clrexne

@ CHECK: clrex                           @ encoding: [0xbf,0xf3,0x2f,0x8f]
@ CHECK: it	ne                       @ encoding: [0x18,0xbf]
@ CHECK: clrexne                         @ encoding: [0xbf,0xf3,0x2f,0x8f]


@------------------------------------------------------------------------------
@ CLZ
@------------------------------------------------------------------------------
        clz r1, r2
        it eq
        clzeq r1, r2

@ CHECK: clz	r1, r2                  @ encoding: [0xb2,0xfa,0x82,0xf1]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: clzeq	r1, r2                  @ encoding: [0xb2,0xfa,0x82,0xf1]


@------------------------------------------------------------------------------
@ CMN
@------------------------------------------------------------------------------
        cmn r1, #0xf
        cmn r8, r6
        cmn r1, r6, lsl #10
        cmn r1, r6, lsr #10
        cmn sp, r6, lsr #10
        cmn r1, r6, asr #10
        cmn r1, r6, ror #10

@ CHECK: cmn.w	r1, #15                 @ encoding: [0x11,0xf1,0x0f,0x0f]
@ CHECK: cmn.w	r8, r6                  @ encoding: [0x18,0xeb,0x06,0x0f]
@ CHECK: cmn.w	r1, r6, lsl #10         @ encoding: [0x11,0xeb,0x86,0x2f]
@ CHECK: cmn.w	r1, r6, lsr #10         @ encoding: [0x11,0xeb,0x96,0x2f]
@ CHECK: cmn.w	sp, r6, lsr #10         @ encoding: [0x1d,0xeb,0x96,0x2f]
@ CHECK: cmn.w	r1, r6, asr #10         @ encoding: [0x11,0xeb,0xa6,0x2f]
@ CHECK: cmn.w	r1, r6, ror #10         @ encoding: [0x11,0xeb,0xb6,0x2f]


@------------------------------------------------------------------------------
@ CMP
@------------------------------------------------------------------------------
        cmp r5, #0xff00
        cmp.w r4, r12
        cmp r9, r6, lsl #12
        cmp r3, r7, lsr #31
        cmp sp, r6, lsr #1
        cmp r2, r5, asr #24
        cmp r1, r4, ror #15
        cmp r2, #-2
        cmp r9, #1

@ CHECK: cmp.w	r5, #65280              @ encoding: [0xb5,0xf5,0x7f,0x4f]
@ CHECK: cmp.w	r4, r12                 @ encoding: [0xb4,0xeb,0x0c,0x0f]
@ CHECK: cmp.w	r9, r6, lsl #12         @ encoding: [0xb9,0xeb,0x06,0x3f]
@ CHECK: cmp.w	r3, r7, lsr #31         @ encoding: [0xb3,0xeb,0xd7,0x7f]
@ CHECK: cmp.w	sp, r6, lsr #1          @ encoding: [0xbd,0xeb,0x56,0x0f]
@ CHECK: cmp.w	r2, r5, asr #24         @ encoding: [0xb2,0xeb,0x25,0x6f]
@ CHECK: cmp.w	r1, r4, ror #15         @ encoding: [0xb1,0xeb,0xf4,0x3f]
@ CHECK: cmn.w	r2, #2                  @ encoding: [0x12,0xf1,0x02,0x0f]
@ CHECK: cmp.w	r9, #1                  @ encoding: [0xb9,0xf1,0x01,0x0f]

@------------------------------------------------------------------------------
@ CPS
@------------------------------------------------------------------------------

        cpsie f
        cpsid a
        cpsie.w f
        cpsid.w a
        cpsie i, #3
        cpsie.w i, #3
        cpsid f, #9
        cpsid.w f, #9
        cps #0
        cps.w #0

@ CHECK: cpsie f                        @ encoding: [0x61,0xb6]
@ CHECK: cpsid a                        @ encoding: [0x74,0xb6]
@ CHECK: cpsie.w f                      @ encoding: [0xaf,0xf3,0x20,0x84]
@ CHECK: cpsid.w a                      @ encoding: [0xaf,0xf3,0x80,0x86]
@ CHECK: cpsie i, #3                    @ encoding: [0xaf,0xf3,0x43,0x85]
@ CHECK: cpsie i, #3                    @ encoding: [0xaf,0xf3,0x43,0x85]
@ CHECK: cpsid f, #9                    @ encoding: [0xaf,0xf3,0x29,0x87]
@ CHECK: cpsid f, #9                    @ encoding: [0xaf,0xf3,0x29,0x87]
@ CHECK: cps   #0                       @ encoding: [0xaf,0xf3,0x00,0x81]
@ CHECK: cps   #0                       @ encoding: [0xaf,0xf3,0x00,0x81]

@------------------------------------------------------------------------------
@ DBG
@------------------------------------------------------------------------------
        dbg #5
        dbg #0
        dbg #15

@ CHECK: dbg	#5                      @ encoding: [0xaf,0xf3,0xf5,0x80]
@ CHECK: dbg	#0                      @ encoding: [0xaf,0xf3,0xf0,0x80]
@ CHECK: dbg	#15                     @ encoding: [0xaf,0xf3,0xff,0x80]


@------------------------------------------------------------------------------
@ DMB
@------------------------------------------------------------------------------
        dmb #0xf
        dmb #0xe
        dmb #0xd
        dmb #0xc
        dmb #0xb
        dmb #0xa
        dmb #0x9
        dmb #0x8
        dmb #0x7
        dmb #0x6
        dmb #0x5
        dmb #0x4
        dmb #0x3
        dmb #0x2
        dmb #0x1
        dmb #0x0

        dmb sy
        dmb st
        dmb sh
        dmb ish
        dmb shst
        dmb ishst
        dmb un
        dmb nsh
        dmb unst
        dmb nshst
        dmb osh
        dmb oshst
        dmb

@ CHECK: dmb	sy                      @ encoding: [0xbf,0xf3,0x5f,0x8f]
@ CHECK: dmb	st                      @ encoding: [0xbf,0xf3,0x5e,0x8f]
@ CHECK: dmb	#0xd                    @ encoding: [0xbf,0xf3,0x5d,0x8f]
@ CHECK: dmb	#0xc                    @ encoding: [0xbf,0xf3,0x5c,0x8f]
@ CHECK: dmb	ish                     @ encoding: [0xbf,0xf3,0x5b,0x8f]
@ CHECK: dmb	ishst                   @ encoding: [0xbf,0xf3,0x5a,0x8f]
@ CHECK: dmb	#0x9                    @ encoding: [0xbf,0xf3,0x59,0x8f]
@ CHECK: dmb	#0x8                    @ encoding: [0xbf,0xf3,0x58,0x8f]
@ CHECK: dmb	nsh                     @ encoding: [0xbf,0xf3,0x57,0x8f]
@ CHECK: dmb	nshst                   @ encoding: [0xbf,0xf3,0x56,0x8f]
@ CHECK: dmb	#0x5                    @ encoding: [0xbf,0xf3,0x55,0x8f]
@ CHECK: dmb	#0x4                    @ encoding: [0xbf,0xf3,0x54,0x8f]
@ CHECK: dmb	osh                     @ encoding: [0xbf,0xf3,0x53,0x8f]
@ CHECK: dmb	oshst                   @ encoding: [0xbf,0xf3,0x52,0x8f]
@ CHECK: dmb	#0x1                    @ encoding: [0xbf,0xf3,0x51,0x8f]
@ CHECK: dmb	#0x0                    @ encoding: [0xbf,0xf3,0x50,0x8f]

@ CHECK: dmb	sy                      @ encoding: [0xbf,0xf3,0x5f,0x8f]
@ CHECK: dmb	st                      @ encoding: [0xbf,0xf3,0x5e,0x8f]
@ CHECK: dmb	ish                     @ encoding: [0xbf,0xf3,0x5b,0x8f]
@ CHECK: dmb	ish                     @ encoding: [0xbf,0xf3,0x5b,0x8f]
@ CHECK: dmb	ishst                   @ encoding: [0xbf,0xf3,0x5a,0x8f]
@ CHECK: dmb	ishst                   @ encoding: [0xbf,0xf3,0x5a,0x8f]
@ CHECK: dmb	nsh                     @ encoding: [0xbf,0xf3,0x57,0x8f]
@ CHECK: dmb	nsh                     @ encoding: [0xbf,0xf3,0x57,0x8f]
@ CHECK: dmb	nshst                   @ encoding: [0xbf,0xf3,0x56,0x8f]
@ CHECK: dmb	nshst                   @ encoding: [0xbf,0xf3,0x56,0x8f]
@ CHECK: dmb	osh                     @ encoding: [0xbf,0xf3,0x53,0x8f]
@ CHECK: dmb	oshst                   @ encoding: [0xbf,0xf3,0x52,0x8f]
@ CHECK: dmb	sy                      @ encoding: [0xbf,0xf3,0x5f,0x8f]


@------------------------------------------------------------------------------
@ DSB
@------------------------------------------------------------------------------
        dsb #0xf
        dsb #0xe
        dsb #0xd
        dsb #0xc
        dsb #0xb
        dsb #0xa
        dsb #0x9
        dsb #0x8
        dsb #0x7
        dsb #0x6
        dsb #0x5
        dsb #0x4
        dsb #0x3
        dsb #0x2
        dsb #0x1
        dsb #0x0

        dsb sy
        dsb st
        dsb sh
        dsb ish
        dsb shst
        dsb ishst
        dsb un
        dsb nsh
        dsb unst
        dsb nshst
        dsb osh
        dsb oshst
        dsb

@ CHECK: dsb	sy                      @ encoding: [0xbf,0xf3,0x4f,0x8f]
@ CHECK: dsb	st                      @ encoding: [0xbf,0xf3,0x4e,0x8f]
@ CHECK: dsb	#0xd                    @ encoding: [0xbf,0xf3,0x4d,0x8f]
@ CHECK: dsb	#0xc                    @ encoding: [0xbf,0xf3,0x4c,0x8f]
@ CHECK: dsb	ish                     @ encoding: [0xbf,0xf3,0x4b,0x8f]
@ CHECK: dsb	ishst                   @ encoding: [0xbf,0xf3,0x4a,0x8f]
@ CHECK: dsb	#0x9                    @ encoding: [0xbf,0xf3,0x49,0x8f]
@ CHECK: dsb	#0x8                    @ encoding: [0xbf,0xf3,0x48,0x8f]
@ CHECK: dsb	nsh                     @ encoding: [0xbf,0xf3,0x47,0x8f]
@ CHECK: dsb	nshst                   @ encoding: [0xbf,0xf3,0x46,0x8f]
@ CHECK: dsb	#0x5                    @ encoding: [0xbf,0xf3,0x45,0x8f]
@ CHECK: pssbb                          @ encoding: [0xbf,0xf3,0x44,0x8f]
@ CHECK: dsb	osh                     @ encoding: [0xbf,0xf3,0x43,0x8f]
@ CHECK: dsb	oshst                   @ encoding: [0xbf,0xf3,0x42,0x8f]
@ CHECK: dsb	#0x1                    @ encoding: [0xbf,0xf3,0x41,0x8f]
@ CHECK: ssbb                           @ encoding: [0xbf,0xf3,0x40,0x8f]

@ CHECK: dsb	sy                      @ encoding: [0xbf,0xf3,0x4f,0x8f]
@ CHECK: dsb	st                      @ encoding: [0xbf,0xf3,0x4e,0x8f]
@ CHECK: dsb	ish                     @ encoding: [0xbf,0xf3,0x4b,0x8f]
@ CHECK: dsb	ish                     @ encoding: [0xbf,0xf3,0x4b,0x8f]
@ CHECK: dsb	ishst                   @ encoding: [0xbf,0xf3,0x4a,0x8f]
@ CHECK: dsb	ishst                   @ encoding: [0xbf,0xf3,0x4a,0x8f]
@ CHECK: dsb	nsh                     @ encoding: [0xbf,0xf3,0x47,0x8f]
@ CHECK: dsb	nsh                     @ encoding: [0xbf,0xf3,0x47,0x8f]
@ CHECK: dsb	nshst                   @ encoding: [0xbf,0xf3,0x46,0x8f]
@ CHECK: dsb	nshst                   @ encoding: [0xbf,0xf3,0x46,0x8f]
@ CHECK: dsb	osh                     @ encoding: [0xbf,0xf3,0x43,0x8f]
@ CHECK: dsb	oshst                   @ encoding: [0xbf,0xf3,0x42,0x8f]
@ CHECK: dsb	sy                      @ encoding: [0xbf,0xf3,0x4f,0x8f]


@------------------------------------------------------------------------------
@ EOR
@------------------------------------------------------------------------------
        eor r4, r5, #0xf000
        eor r4, r5, r6
        eor r4, r5, r6, lsl #5
        eor r4, r5, r6, lsr #5
        eor r4, r5, r6, lsr #5
        eor r4, r5, r6, asr #5
        eor r4, r5, r6, ror #5

@ CHECK: eor	r4, r5, #61440          @ encoding: [0x85,0xf4,0x70,0x44]
@ CHECK: eor.w	r4, r5, r6              @ encoding: [0x85,0xea,0x06,0x04]
@ CHECK: eor.w	r4, r5, r6, lsl #5      @ encoding: [0x85,0xea,0x46,0x14]
@ CHECK: eor.w	r4, r5, r6, lsr #5      @ encoding: [0x85,0xea,0x56,0x14]
@ CHECK: eor.w	r4, r5, r6, lsr #5      @ encoding: [0x85,0xea,0x56,0x14]
@ CHECK: eor.w	r4, r5, r6, asr #5      @ encoding: [0x85,0xea,0x66,0x14]
@ CHECK: eor.w	r4, r5, r6, ror #5      @ encoding: [0x85,0xea,0x76,0x14]


@------------------------------------------------------------------------------
@ ISB
@------------------------------------------------------------------------------
        isb sy
        isb
        isb #15
        isb #1

@ CHECK: isb	sy                      @ encoding: [0xbf,0xf3,0x6f,0x8f]
@ CHECK: isb	sy                      @ encoding: [0xbf,0xf3,0x6f,0x8f]
@ CHECK: isb	sy                      @ encoding: [0xbf,0xf3,0x6f,0x8f]
@ CHECK: isb	#0x1                    @ encoding: [0xbf,0xf3,0x61,0x8f]


@------------------------------------------------------------------------------
@ IT
@------------------------------------------------------------------------------
@ Test encodings of a few full IT blocks, not just the IT instruction

        iteet eq
        addeq r0, r1, r2
        nopne
        subne r5, r6, r7
        addeq r1, r2, #4

@ CHECK: iteet	eq                      @ encoding: [0x0d,0xbf]
@ CHECK: addeq	r0, r1, r2              @ encoding: [0x88,0x18]
@ CHECK: nopne                          @ encoding: [0x00,0xbf]
@ CHECK: subne	r5, r6, r7              @ encoding: [0xf5,0x1b]
@ CHECK: addeq	r1, r2, #4              @ encoding: [0x11,0x1d]

@ Should also work for UPPER CASE condition codes.

        ITEET EQ
        ADDEQ R0, R1, R2
        NOPNE
        SUBNE R5, R6, R7
        ADDEQ R1, R2, #4

@ CHECK: iteet	eq                      @ encoding: [0x0d,0xbf]
@ CHECK: addeq	r0, r1, r2              @ encoding: [0x88,0x18]
@ CHECK: nopne                          @ encoding: [0x00,0xbf]
@ CHECK: subne	r5, r6, r7              @ encoding: [0xf5,0x1b]
@ CHECK: addeq	r1, r2, #4              @ encoding: [0x11,0x1d]

@------------------------------------------------------------------------------
@ LDC{L}/LDC2{L}
@------------------------------------------------------------------------------
        ldc2 p0, c8, [r1, #4]
        ldc2 p1, c7, [r2]
        ldc2 p2, c6, [r3, #-224]
        ldc2 p3, c5, [r4, #-120]!
        ldc2 p4, c4, [r5], #16
        ldc2 p5, c3, [r6], #-72
        ldc2l p6, c2, [r7, #4]
        ldc2l p7, c1, [r8]
        ldc2l p8, c0, [r9, #-224]
        ldc2l p9, c1, [r10, #-120]!
        ldc2l p0, c2, [r11], #16
        ldc2l p1, c3, [r12], #-72

        ldc p12, c4, [r0, #4]
        ldc p13, c5, [r1]
        ldc p14, c6, [r2, #-224]
        ldc p15, c7, [r3, #-120]!
        ldc p5, c8, [r4], #16
        ldc p4, c9, [r5], #-72
        ldcl p3, c10, [r6, #4]
        ldcl p2, c11, [r7]
        ldcl p1, c12, [r8, #-224]
        ldcl p0, c13, [r9, #-120]!
        ldcl p6, c14, [r10], #16
        ldcl p7, c15, [r11], #-72

        ldc2 p2, c8, [r1], { 25 }

@ CHECK: ldc2	p0, c8, [r1, #4]        @ encoding: [0x91,0xfd,0x01,0x80]
@ CHECK: ldc2	p1, c7, [r2]            @ encoding: [0x92,0xfd,0x00,0x71]
@ CHECK: ldc2	p2, c6, [r3, #-224]     @ encoding: [0x13,0xfd,0x38,0x62]
@ CHECK: ldc2	p3, c5, [r4, #-120]!    @ encoding: [0x34,0xfd,0x1e,0x53]
@ CHECK: ldc2	p4, c4, [r5], #16       @ encoding: [0xb5,0xfc,0x04,0x44]
@ CHECK: ldc2	p5, c3, [r6], #-72      @ encoding: [0x36,0xfc,0x12,0x35]
@ CHECK: ldc2l	p6, c2, [r7, #4]        @ encoding: [0xd7,0xfd,0x01,0x26]
@ CHECK: ldc2l	p7, c1, [r8]            @ encoding: [0xd8,0xfd,0x00,0x17]
@ CHECK: ldc2l	p8, c0, [r9, #-224]     @ encoding: [0x59,0xfd,0x38,0x08]
@ CHECK: ldc2l	p9, c1, [r10, #-120]!   @ encoding: [0x7a,0xfd,0x1e,0x19]
@ CHECK: ldc2l	p0, c2, [r11], #16      @ encoding: [0xfb,0xfc,0x04,0x20]
@ CHECK: ldc2l	p1, c3, [r12], #-72     @ encoding: [0x7c,0xfc,0x12,0x31]

@ CHECK: ldc	p12, c4, [r0, #4]       @ encoding: [0x90,0xed,0x01,0x4c]
@ CHECK: ldc	p13, c5, [r1]           @ encoding: [0x91,0xed,0x00,0x5d]
@ CHECK: ldc	p14, c6, [r2, #-224]    @ encoding: [0x12,0xed,0x38,0x6e]
@ CHECK: ldc	p15, c7, [r3, #-120]!   @ encoding: [0x33,0xed,0x1e,0x7f]
@ CHECK: ldc	p5, c8, [r4], #16       @ encoding: [0xb4,0xec,0x04,0x85]
@ CHECK: ldc	p4, c9, [r5], #-72      @ encoding: [0x35,0xec,0x12,0x94]
@ CHECK: ldcl	p3, c10, [r6, #4]       @ encoding: [0xd6,0xed,0x01,0xa3]
@ CHECK: ldcl	p2, c11, [r7]           @ encoding: [0xd7,0xed,0x00,0xb2]
@ CHECK: ldcl	p1, c12, [r8, #-224]    @ encoding: [0x58,0xed,0x38,0xc1]
@ CHECK: ldcl	p0, c13, [r9, #-120]!   @ encoding: [0x79,0xed,0x1e,0xd0]
@ CHECK: ldcl	p6, c14, [r10], #16     @ encoding: [0xfa,0xec,0x04,0xe6]
@ CHECK: ldcl	p7, c15, [r11], #-72    @ encoding: [0x7b,0xec,0x12,0xf7]

@ CHECK: ldc2	p2, c8, [r1], {25}      @ encoding: [0x91,0xfc,0x19,0x82]


@------------------------------------------------------------------------------
@ LDMIA
@------------------------------------------------------------------------------
        ldmia.w r4, {r4, r5, r8, r9}
        ldmia.w r4, {r5, r6}
        ldmia.w r5!, {r3, r8}
        ldm.w r4, {r4, r5, r8, r9}
        ldm.w r4, {r5, r6}
        ldm.w r5!, {r3, r8}
        ldm.w r5!, {r1, r2}
        ldm.w r2, {r1, r2}

        ldmia r4, {r4, r5, r8, r9}
        ldmia r4, {r5, r6}
        ldmia r5!, {r3, r8}
        ldm r4, {r4, r5, r8, r9}
        ldm r4, {r5, r6}
        ldm r5!, {r3, r8}
        ldmfd r5!, {r3, r8}
        ldmia sp!, {r4-r11, pc}

@ CHECK: ldm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x94,0xe8,0x30,0x03]
@ CHECK: ldm.w	r4, {r5, r6}            @ encoding: [0x94,0xe8,0x60,0x00]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]
@ CHECK: ldm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x94,0xe8,0x30,0x03]
@ CHECK: ldm.w	r4, {r5, r6}            @ encoding: [0x94,0xe8,0x60,0x00]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]
@ CHECK: ldm.w	r5!, {r1, r2}           @ encoding: [0xb5,0xe8,0x06,0x00]
@ CHECK: ldm.w	r2, {r1, r2}            @ encoding: [0x92,0xe8,0x06,0x00]

@ CHECK: ldm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x94,0xe8,0x30,0x03]
@ CHECK: ldm.w	r4, {r5, r6}            @ encoding: [0x94,0xe8,0x60,0x00]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]
@ CHECK: ldm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x94,0xe8,0x30,0x03]
@ CHECK: ldm.w	r4, {r5, r6}            @ encoding: [0x94,0xe8,0x60,0x00]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]
@ CHECK: ldm.w	r5!, {r3, r8}           @ encoding: [0xb5,0xe8,0x08,0x01]
@ CHECK: pop.w	{r4, r5, r6, r7, r8, r9, r10, r11, pc} @ encoding: [0xbd,0xe8,0xf0,0x8f]


@------------------------------------------------------------------------------
@ LDMDB
@------------------------------------------------------------------------------
        ldmdb r4, {r4, r5, r8, r9}
        ldmdb r4, {r5, r6}
        ldmdb r5!, {r3, r8}
        ldmea r5!, {r3, r8}
        ldmdb.w r4, {r5, r6}
        ldmdb.w r5!, {r3, r8}

@ CHECK: ldmdb	r4, {r4, r5, r8, r9}    @ encoding: [0x14,0xe9,0x30,0x03]
@ CHECK: ldmdb	r4, {r5, r6}            @ encoding: [0x14,0xe9,0x60,0x00]
@ CHECK: ldmdb	r5!, {r3, r8}           @ encoding: [0x35,0xe9,0x08,0x01]
@ CHECK: ldmdb	r5!, {r3, r8}           @ encoding: [0x35,0xe9,0x08,0x01]
@ CHECK: ldmdb	r4, {r5, r6}            @ encoding: [0x14,0xe9,0x60,0x00]
@ CHECK: ldmdb	r5!, {r3, r8}           @ encoding: [0x35,0xe9,0x08,0x01]


@------------------------------------------------------------------------------
@ LDR(immediate)
@------------------------------------------------------------------------------
        ldr r5, [r5, #-4]
        ldr r5, [r6, #32]
        ldr r5, [r6, #33]
        ldr r5, [r6, #257]
        ldr.w pc, [r7, #257]
        ldr r2, [r4, #255]!
        ldr r8, [sp, #4]!
        ldr lr, [sp, #-4]!
        ldr r2, [r4], #255
        ldr r8, [sp], #4
        ldr lr, [sp], #-4

@ CHECK: ldr	r5, [r5, #-4]           @ encoding: [0x55,0xf8,0x04,0x5c]
@ CHECK: ldr	r5, [r6, #32]           @ encoding: [0x35,0x6a]
@ CHECK: ldr.w	r5, [r6, #33]           @ encoding: [0xd6,0xf8,0x21,0x50]
@ CHECK: ldr.w	r5, [r6, #257]          @ encoding: [0xd6,0xf8,0x01,0x51]
@ CHECK: ldr.w	pc, [r7, #257]          @ encoding: [0xd7,0xf8,0x01,0xf1]
@ CHECK: ldr	r2, [r4, #255]!         @ encoding: [0x54,0xf8,0xff,0x2f]
@ CHECK: ldr	r8, [sp, #4]!           @ encoding: [0x5d,0xf8,0x04,0x8f]
@ CHECK: ldr	lr, [sp, #-4]!          @ encoding: [0x5d,0xf8,0x04,0xed]
@ CHECK: ldr	r2, [r4], #255          @ encoding: [0x54,0xf8,0xff,0x2b]
@ CHECK: ldr	r8, [sp], #4            @ encoding: [0x5d,0xf8,0x04,0x8b]
@ CHECK: ldr	lr, [sp], #-4           @ encoding: [0x5d,0xf8,0x04,0xe9]


@------------------------------------------------------------------------------
@ LDR(literal)
@------------------------------------------------------------------------------
        ldr.w r5, _foo
        ldr   lr, (_strcmp-4)
        ldr sp, _foo
        ldr pc, _foo

@ CHECK: ldr.w	r5, _foo                @ encoding: [0x5f'A',0xf8'A',A,0x50'A']
@ CHECK: @   fixup A - offset: 0, value: _foo, kind: fixup_t2_ldst_pcrel_12
@ CHECK-BE: ldr.w	r5, _foo                @ encoding: [0xf8'A',0x5f'A',0x50'A',A]
@ CHECK-BE: @   fixup A - offset: 0, value: _foo, kind: fixup_t2_ldst_pcrel_12
@ CHECK: ldr.w	lr, _strcmp-4           @ encoding: [0x5f'A',0xf8'A',A,0xe0'A']
@ CHECK: @   fixup A - offset: 0, value: _strcmp-4, kind: fixup_t2_ldst_pcrel_12
@ CHECK-BE: ldr.w	lr, _strcmp-4           @ encoding: [0xf8'A',0x5f'A',0xe0'A',A]
@ CHECK-BE: @   fixup A - offset: 0, value: _strcmp-4, kind: fixup_t2_ldst_pcrel_12
@ CHECK: ldr.w sp, _foo                 @ encoding: [0x5f'A',0xf8'A',A,0xd0'A']
@ CHECK: @   fixup A - offset: 0, value: _foo, kind: fixup_t2_ldst_pcrel_12
@ CHECK-BE: ldr.w sp, _foo                 @ encoding: [0xf8'A',0x5f'A',0xd0'A',A]
@ CHECK-BE: @   fixup A - offset: 0, value: _foo, kind: fixup_t2_ldst_pcrel_12
@ CHECK: ldr.w pc, _foo                 @ encoding: [0x5f'A',0xf8'A',A,0xf0'A']
@ CHECK: @   fixup A - offset: 0, value: _foo, kind: fixup_t2_ldst_pcrel_12
@ CHECK-BE: ldr.w pc, _foo                 @ encoding: [0xf8'A',0x5f'A',0xf0'A',A]
@ CHECK-BE: @   fixup A - offset: 0, value: _foo, kind: fixup_t2_ldst_pcrel_12

        ldr r7, [pc, #8]
        ldr.n r7, [pc, #8]
        ldr.w r7, [pc, #8]
        ldr r4, [pc, #1020]
        ldr r3, [pc, #-1020]
        ldr r6, [pc, #1024]
        ldr r0, [pc, #-1024]
        ldr r2, [pc, #4095]
        ldr r1, [pc, #-4095]
        ldr r8, [pc, #132]
        ldr pc, [pc, #256]
        ldr pc, [pc, #-400]
        ldr sp, [pc, #4]

@ CHECK: ldr	r7, [pc, #8]            @ encoding: [0x02,0x4f]
@ CHECK: ldr	r7, [pc, #8]            @ encoding: [0x02,0x4f]
@ CHECK: ldr.w	r7, [pc, #8]            @ encoding: [0xdf,0xf8,0x08,0x70]
@ CHECK: ldr	r4, [pc, #1020]         @ encoding: [0xff,0x4c]
@ CHECK: ldr.w	r3, [pc, #-1020]        @ encoding: [0x5f,0xf8,0xfc,0x33]
@ CHECK: ldr.w	r6, [pc, #1024]         @ encoding: [0xdf,0xf8,0x00,0x64]
@ CHECK: ldr.w	r0, [pc, #-1024]        @ encoding: [0x5f,0xf8,0x00,0x04]
@ CHECK: ldr.w	r2, [pc, #4095]         @ encoding: [0xdf,0xf8,0xff,0x2f]
@ CHECK: ldr.w	r1, [pc, #-4095]        @ encoding: [0x5f,0xf8,0xff,0x1f]
@ CHECK: ldr.w	r8, [pc, #132]          @ encoding: [0xdf,0xf8,0x84,0x80]
@ CHECK: ldr.w	pc, [pc, #256]          @ encoding: [0xdf,0xf8,0x00,0xf1]
@ CHECK: ldr.w	pc, [pc, #-400]         @ encoding: [0x5f,0xf8,0x90,0xf1]
@ CHECK: ldr.w  sp, [pc, #4]            @ encoding: [0xdf,0xf8,0x04,0xd0]

        ldrb  r9, [pc, #-0]
        ldrsb r11, [pc, #-0]
        ldrh  r10, [pc, #-0]
        ldrsh r1, [pc, #-0]
        ldr   r5, [pc, #-0]

@ CHECK: ldrb.w	r9, [pc, #-0]           @ encoding: [0x1f,0xf8,0x00,0x90]
@ CHECK: ldrsb.w	r11, [pc, #-0]  @ encoding: [0x1f,0xf9,0x00,0xb0]
@ CHECK: ldrh.w	r10, [pc, #-0]          @ encoding: [0x3f,0xf8,0x00,0xa0]
@ CHECK: ldrsh.w	r1, [pc, #-0]   @ encoding: [0x3f,0xf9,0x00,0x10]
@ CHECK: ldr.w	r5, [pc, #-0]           @ encoding: [0x5f,0xf8,0x00,0x50]

@------------------------------------------------------------------------------
@ LDR(register)
@------------------------------------------------------------------------------
        ldr r1, [r8, r1]
        ldr.w r4, [r5, r2]
        ldr r6, [r0, r2, lsl #3]
        ldr r8, [r8, r2, lsl #2]
        ldr r7, [sp, r2, lsl #1]
        ldr r7, [sp, r2, lsl #0]

@ CHECK: ldr.w	r1, [r8, r1]            @ encoding: [0x58,0xf8,0x01,0x10]
@ CHECK: ldr.w	r4, [r5, r2]            @ encoding: [0x55,0xf8,0x02,0x40]
@ CHECK: ldr.w	r6, [r0, r2, lsl #3]    @ encoding: [0x50,0xf8,0x32,0x60]
@ CHECK: ldr.w	r8, [r8, r2, lsl #2]    @ encoding: [0x58,0xf8,0x22,0x80]
@ CHECK: ldr.w	r7, [sp, r2, lsl #1]    @ encoding: [0x5d,0xf8,0x12,0x70]
@ CHECK: ldr.w	r7, [sp, r2]            @ encoding: [0x5d,0xf8,0x02,0x70]


@------------------------------------------------------------------------------
@ LDRB(immediate)
@------------------------------------------------------------------------------
        ldrb r5, [r5, #-4]
        ldrb r5, [r6, #32]
        ldrb r5, [r6, #33]
        ldrb r5, [r6, #257]
        ldrb.w lr, [r7, #257]
        ldrb r5, [r8, #255]!
        ldrb r2, [r5, #4]!
        ldrb r1, [r4, #-4]!
        ldrb lr, [r3], #255
        ldrb r9, [r2], #4
        ldrb r3, [sp], #-4

@ CHECK: ldrb	r5, [r5, #-4]           @ encoding: [0x15,0xf8,0x04,0x5c]
@ CHECK: ldrb.w	r5, [r6, #32]           @ encoding: [0x96,0xf8,0x20,0x50]
@ CHECK: ldrb.w	r5, [r6, #33]           @ encoding: [0x96,0xf8,0x21,0x50]
@ CHECK: ldrb.w	r5, [r6, #257]          @ encoding: [0x96,0xf8,0x01,0x51]
@ CHECK: ldrb.w	lr, [r7, #257]          @ encoding: [0x97,0xf8,0x01,0xe1]
@ CHECK: ldrb	r5, [r8, #255]!         @ encoding: [0x18,0xf8,0xff,0x5f]
@ CHECK: ldrb	r2, [r5, #4]!           @ encoding: [0x15,0xf8,0x04,0x2f]
@ CHECK: ldrb	r1, [r4, #-4]!          @ encoding: [0x14,0xf8,0x04,0x1d]
@ CHECK: ldrb	lr, [r3], #255          @ encoding: [0x13,0xf8,0xff,0xeb]
@ CHECK: ldrb	r9, [r2], #4            @ encoding: [0x12,0xf8,0x04,0x9b]
@ CHECK: ldrb	r3, [sp], #-4           @ encoding: [0x1d,0xf8,0x04,0x39]


@------------------------------------------------------------------------------
@ LDRB(register)
@------------------------------------------------------------------------------
        ldrb r1, [r8, r1]
        ldrb.w r4, [r5, r2]
        ldrb r6, [r0, r2, lsl #3]
        ldrb r8, [r8, r2, lsl #2]
        ldrb r7, [sp, r2, lsl #1]
        ldrb r7, [sp, r2, lsl #0]

@ CHECK: ldrb.w	r1, [r8, r1]            @ encoding: [0x18,0xf8,0x01,0x10]
@ CHECK: ldrb.w	r4, [r5, r2]            @ encoding: [0x15,0xf8,0x02,0x40]
@ CHECK: ldrb.w	r6, [r0, r2, lsl #3]    @ encoding: [0x10,0xf8,0x32,0x60]
@ CHECK: ldrb.w	r8, [r8, r2, lsl #2]    @ encoding: [0x18,0xf8,0x22,0x80]
@ CHECK: ldrb.w	r7, [sp, r2, lsl #1]    @ encoding: [0x1d,0xf8,0x12,0x70]
@ CHECK: ldrb.w	r7, [sp, r2]            @ encoding: [0x1d,0xf8,0x02,0x70]


@------------------------------------------------------------------------------
@ LDRBT
@------------------------------------------------------------------------------
        ldrbt r1, [r2]
        ldrbt r1, [r8, #0]
        ldrbt r1, [r8, #3]
        ldrbt r1, [r8, #255]

@ CHECK: ldrbt	r1, [r2]                @ encoding: [0x12,0xf8,0x00,0x1e]
@ CHECK: ldrbt	r1, [r8]                @ encoding: [0x18,0xf8,0x00,0x1e]
@ CHECK: ldrbt	r1, [r8, #3]            @ encoding: [0x18,0xf8,0x03,0x1e]
@ CHECK: ldrbt	r1, [r8, #255]          @ encoding: [0x18,0xf8,0xff,0x1e]


@------------------------------------------------------------------------------
@ LDRD
@------------------------------------------------------------------------------
        ldrd r3, r5, [r6, #24]
        ldrd r3, r5, [r6, #24]!
        ldrd r3, r5, [r6], #4
        ldrd r3, r5, [r6], #-8
        ldrd r3, r5, [r6]
        ldrd r8, r1, [r3, #0]
        ldrd r0, r1, [r2, #-0]
        ldrd r0, r1, [r2, #-0]!
        ldrd r0, r1, [r2], #-0

@ CHECK: ldrd	r3, r5, [r6, #24]       @ encoding: [0xd6,0xe9,0x06,0x35]
@ CHECK: ldrd	r3, r5, [r6, #24]!      @ encoding: [0xf6,0xe9,0x06,0x35]
@ CHECK: ldrd	r3, r5, [r6], #4        @ encoding: [0xf6,0xe8,0x01,0x35]
@ CHECK: ldrd	r3, r5, [r6], #-8       @ encoding: [0x76,0xe8,0x02,0x35]
@ CHECK: ldrd	r3, r5, [r6]            @ encoding: [0xd6,0xe9,0x00,0x35]
@ CHECK: ldrd	r8, r1, [r3]            @ encoding: [0xd3,0xe9,0x00,0x81]
@ CHECK: ldrd	r0, r1, [r2, #-0]       @ encoding: [0x52,0xe9,0x00,0x01]
@ CHECK: ldrd	r0, r1, [r2, #-0]!      @ encoding: [0x72,0xe9,0x00,0x01]
@ CHECK: ldrd	r0, r1, [r2], #-0       @ encoding: [0x72,0xe8,0x00,0x01]


@------------------------------------------------------------------------------
@ FIXME: LDRD(literal)
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ LDREX/LDREXB/LDREXH/LDREXD
@------------------------------------------------------------------------------
        ldrex r1, [r4]
        ldrex r8, [r4, #0]
        ldrex r2, [sp, #128]
        ldrexb r5, [r7]
        ldrexh r9, [r12]
        ldrexd r9, r3, [r4]

@ CHECK: ldrex	r1, [r4]                @ encoding: [0x54,0xe8,0x00,0x1f]
@ CHECK: ldrex	r8, [r4]                @ encoding: [0x54,0xe8,0x00,0x8f]
@ CHECK: ldrex	r2, [sp, #128]          @ encoding: [0x5d,0xe8,0x20,0x2f]
@ CHECK: ldrexb	r5, [r7]                @ encoding: [0xd7,0xe8,0x4f,0x5f]
@ CHECK: ldrexh	r9, [r12]               @ encoding: [0xdc,0xe8,0x5f,0x9f]
@ CHECK: ldrexd	r9, r3, [r4]            @ encoding: [0xd4,0xe8,0x7f,0x93]


@------------------------------------------------------------------------------
@ LDRH(immediate)
@------------------------------------------------------------------------------
        ldrh r5, [r5, #-4]
        ldrh r5, [r6, #32]
        ldrh r5, [r6, #33]
        ldrh r5, [r6, #257]
        ldrh.w lr, [r7, #257]
        ldrh r5, [r8, #255]!
        ldrh r2, [r5, #4]!
        ldrh r1, [r4, #-4]!
        ldrh lr, [r3], #255
        ldrh r9, [r2], #4
        ldrh r3, [sp], #-4

@ CHECK: ldrh	r5, [r5, #-4]           @ encoding: [0x35,0xf8,0x04,0x5c]
@ CHECK: ldrh	r5, [r6, #32]           @ encoding: [0x35,0x8c]
@ CHECK: ldrh.w	r5, [r6, #33]           @ encoding: [0xb6,0xf8,0x21,0x50]
@ CHECK: ldrh.w	r5, [r6, #257]          @ encoding: [0xb6,0xf8,0x01,0x51]
@ CHECK: ldrh.w	lr, [r7, #257]          @ encoding: [0xb7,0xf8,0x01,0xe1]
@ CHECK: ldrh	r5, [r8, #255]!         @ encoding: [0x38,0xf8,0xff,0x5f]
@ CHECK: ldrh	r2, [r5, #4]!           @ encoding: [0x35,0xf8,0x04,0x2f]
@ CHECK: ldrh	r1, [r4, #-4]!          @ encoding: [0x34,0xf8,0x04,0x1d]
@ CHECK: ldrh	lr, [r3], #255          @ encoding: [0x33,0xf8,0xff,0xeb]
@ CHECK: ldrh	r9, [r2], #4            @ encoding: [0x32,0xf8,0x04,0x9b]
@ CHECK: ldrh	r3, [sp], #-4           @ encoding: [0x3d,0xf8,0x04,0x39]


@------------------------------------------------------------------------------
@ LDRH(register)
@------------------------------------------------------------------------------
        ldrh r1, [r8, r1]
        ldrh.w r4, [r5, r2]
        ldrh r6, [r0, r2, lsl #3]
        ldrh r8, [r8, r2, lsl #2]
        ldrh r7, [sp, r2, lsl #1]
        ldrh r7, [sp, r2, lsl #0]

@ CHECK: ldrh.w	r1, [r8, r1]            @ encoding: [0x38,0xf8,0x01,0x10]
@ CHECK: ldrh.w	r4, [r5, r2]            @ encoding: [0x35,0xf8,0x02,0x40]
@ CHECK: ldrh.w	r6, [r0, r2, lsl #3]    @ encoding: [0x30,0xf8,0x32,0x60]
@ CHECK: ldrh.w	r8, [r8, r2, lsl #2]    @ encoding: [0x38,0xf8,0x22,0x80]
@ CHECK: ldrh.w	r7, [sp, r2, lsl #1]    @ encoding: [0x3d,0xf8,0x12,0x70]
@ CHECK: ldrh.w	r7, [sp, r2]            @ encoding: [0x3d,0xf8,0x02,0x70]


@------------------------------------------------------------------------------
@ LDRH(literal)
@------------------------------------------------------------------------------
        ldrh r5, _bar

@ CHECK: ldrh.w	r5, _bar                @ encoding: [0x3f'A',0xf8'A',A,0x50'A']
@ CHECK:     @   fixup A - offset: 0, value: _bar, kind: fixup_t2_ldst_pcrel_12
@ CHECK-BE: ldrh.w	r5, _bar                @ encoding: [0xf8'A',0x3f'A',0x50'A',A]
@ CHECK-BE:     @   fixup A - offset: 0, value: _bar, kind: fixup_t2_ldst_pcrel_12


@------------------------------------------------------------------------------
@ LDRHT
@------------------------------------------------------------------------------
        ldrht r1, [r2]
        ldrht r1, [r8, #0]
        ldrht r1, [r8, #3]
        ldrht r1, [r8, #255]

@ CHECK: ldrht	r1, [r2]                @ encoding: [0x32,0xf8,0x00,0x1e]
@ CHECK: ldrht	r1, [r8]                @ encoding: [0x38,0xf8,0x00,0x1e]
@ CHECK: ldrht	r1, [r8, #3]            @ encoding: [0x38,0xf8,0x03,0x1e]
@ CHECK: ldrht	r1, [r8, #255]          @ encoding: [0x38,0xf8,0xff,0x1e]


@------------------------------------------------------------------------------
@ LDRSB(immediate)
@------------------------------------------------------------------------------
        ldrsb r5, [r5, #-4]
        ldrsb r5, [r6, #32]
        ldrsb r5, [r6, #33]
        ldrsb r5, [r6, #257]
        ldrsb.w lr, [r7, #257]

@ CHECK: ldrsb	r5, [r5, #-4]            @ encoding: [0x15,0xf9,0x04,0x5c]
@ CHECK: ldrsb.w r5, [r6, #32]           @ encoding: [0x96,0xf9,0x20,0x50]
@ CHECK: ldrsb.w r5, [r6, #33]           @ encoding: [0x96,0xf9,0x21,0x50]
@ CHECK: ldrsb.w r5, [r6, #257]          @ encoding: [0x96,0xf9,0x01,0x51]
@ CHECK: ldrsb.w lr, [r7, #257]          @ encoding: [0x97,0xf9,0x01,0xe1]


@------------------------------------------------------------------------------
@ LDRSB(register)
@------------------------------------------------------------------------------
        ldrsb r1, [r8, r1]
        ldrsb.w r4, [r5, r2]
        ldrsb r6, [r0, r2, lsl #3]
        ldrsb r8, [r8, r2, lsl #2]
        ldrsb r7, [sp, r2, lsl #1]
        ldrsb r7, [sp, r2, lsl #0]
        ldrsb r5, [r8, #255]!
        ldrsb r2, [r5, #4]!
        ldrsb r1, [r4, #-4]!
        ldrsb lr, [r3], #255
        ldrsb r9, [r2], #4
        ldrsb r3, [sp], #-4

@ CHECK: ldrsb.w r1, [r8, r1]           @ encoding: [0x18,0xf9,0x01,0x10]
@ CHECK: ldrsb.w r4, [r5, r2]           @ encoding: [0x15,0xf9,0x02,0x40]
@ CHECK: ldrsb.w r6, [r0, r2, lsl #3]   @ encoding: [0x10,0xf9,0x32,0x60]
@ CHECK: ldrsb.w r8, [r8, r2, lsl #2]   @ encoding: [0x18,0xf9,0x22,0x80]
@ CHECK: ldrsb.w r7, [sp, r2, lsl #1]   @ encoding: [0x1d,0xf9,0x12,0x70]
@ CHECK: ldrsb.w r7, [sp, r2]           @ encoding: [0x1d,0xf9,0x02,0x70]
@ CHECK: ldrsb	r5, [r8, #255]!         @ encoding: [0x18,0xf9,0xff,0x5f]
@ CHECK: ldrsb	r2, [r5, #4]!           @ encoding: [0x15,0xf9,0x04,0x2f]
@ CHECK: ldrsb	r1, [r4, #-4]!          @ encoding: [0x14,0xf9,0x04,0x1d]
@ CHECK: ldrsb	lr, [r3], #255          @ encoding: [0x13,0xf9,0xff,0xeb]
@ CHECK: ldrsb	r9, [r2], #4            @ encoding: [0x12,0xf9,0x04,0x9b]
@ CHECK: ldrsb	r3, [sp], #-4           @ encoding: [0x1d,0xf9,0x04,0x39]


@------------------------------------------------------------------------------
@ LDRSB(literal)
@------------------------------------------------------------------------------
        ldrsb r5, _bar

@ CHECK: ldrsb.w r5, _bar               @ encoding: [0x1f'A',0xf9'A',A,0x50'A']
@ CHECK:      @   fixup A - offset: 0, value: _bar, kind: fixup_t2_ldst_pcrel_12
@ CHECK-BE: ldrsb.w r5, _bar               @ encoding: [0xf9'A',0x1f'A',0x50'A',A]
@ CHECK-BE:      @   fixup A - offset: 0, value: _bar, kind: fixup_t2_ldst_pcrel_12


@------------------------------------------------------------------------------
@ LDRSBT
@------------------------------------------------------------------------------
        ldrsbt r1, [r2]
        ldrsbt r1, [r8, #0]
        ldrsbt r1, [r8, #3]
        ldrsbt r1, [r8, #255]

@ CHECK: ldrsbt	r1, [r2]                @ encoding: [0x12,0xf9,0x00,0x1e]
@ CHECK: ldrsbt	r1, [r8]                @ encoding: [0x18,0xf9,0x00,0x1e]
@ CHECK: ldrsbt	r1, [r8, #3]            @ encoding: [0x18,0xf9,0x03,0x1e]
@ CHECK: ldrsbt	r1, [r8, #255]          @ encoding: [0x18,0xf9,0xff,0x1e]


@------------------------------------------------------------------------------
@ LDRSH(immediate)
@------------------------------------------------------------------------------
        ldrsh r5, [r5, #-4]
        ldrsh r5, [r6, #32]
        ldrsh r5, [r6, #33]
        ldrsh r5, [r6, #257]
        ldrsh.w lr, [r7, #257]

@ CHECK: ldrsh	r5, [r5, #-4]           @ encoding: [0x35,0xf9,0x04,0x5c]
@ CHECK: ldrsh.w r5, [r6, #32]          @ encoding: [0xb6,0xf9,0x20,0x50]
@ CHECK: ldrsh.w r5, [r6, #33]          @ encoding: [0xb6,0xf9,0x21,0x50]
@ CHECK: ldrsh.w r5, [r6, #257]         @ encoding: [0xb6,0xf9,0x01,0x51]
@ CHECK: ldrsh.w lr, [r7, #257]         @ encoding: [0xb7,0xf9,0x01,0xe1]


@------------------------------------------------------------------------------
@ LDRSH(register)
@------------------------------------------------------------------------------
        ldrsh r1, [r8, r1]
        ldrsh.w r4, [r5, r2]
        ldrsh r6, [r0, r2, lsl #3]
        ldrsh r8, [r8, r2, lsl #2]
        ldrsh r7, [sp, r2, lsl #1]
        ldrsh r7, [sp, r2, lsl #0]
        ldrsh r5, [r8, #255]!
        ldrsh r2, [r5, #4]!
        ldrsh r1, [r4, #-4]!
        ldrsh lr, [r3], #255
        ldrsh r9, [r2], #4
        ldrsh r3, [sp], #-4

@ CHECK: ldrsh.w r1, [r8, r1]           @ encoding: [0x38,0xf9,0x01,0x10]
@ CHECK: ldrsh.w r4, [r5, r2]           @ encoding: [0x35,0xf9,0x02,0x40]
@ CHECK: ldrsh.w r6, [r0, r2, lsl #3]   @ encoding: [0x30,0xf9,0x32,0x60]
@ CHECK: ldrsh.w r8, [r8, r2, lsl #2]   @ encoding: [0x38,0xf9,0x22,0x80]
@ CHECK: ldrsh.w r7, [sp, r2, lsl #1]   @ encoding: [0x3d,0xf9,0x12,0x70]
@ CHECK: ldrsh.w r7, [sp, r2]           @ encoding: [0x3d,0xf9,0x02,0x70]
@ CHECK: ldrsh	r5, [r8, #255]!         @ encoding: [0x38,0xf9,0xff,0x5f]
@ CHECK: ldrsh	r2, [r5, #4]!           @ encoding: [0x35,0xf9,0x04,0x2f]
@ CHECK: ldrsh	r1, [r4, #-4]!          @ encoding: [0x34,0xf9,0x04,0x1d]
@ CHECK: ldrsh	lr, [r3], #255          @ encoding: [0x33,0xf9,0xff,0xeb]
@ CHECK: ldrsh	r9, [r2], #4            @ encoding: [0x32,0xf9,0x04,0x9b]
@ CHECK: ldrsh	r3, [sp], #-4           @ encoding: [0x3d,0xf9,0x04,0x39]


@------------------------------------------------------------------------------
@ LDRSH(literal)
@------------------------------------------------------------------------------
        ldrsh r5, _bar

@ CHECK: ldrsh.w r5, _bar               @ encoding: [0x3f'A',0xf9'A',A,0x50'A']
@ CHECK:      @   fixup A - offset: 0, value: _bar, kind: fixup_t2_ldst_pcrel_12
@ CHECK-BE: ldrsh.w r5, _bar               @ encoding: [0xf9'A',0x3f'A',0x50'A',A]
@ CHECK-BE:      @   fixup A - offset: 0, value: _bar, kind: fixup_t2_ldst_pcrel_12

@ TEMPORARILY DISABLED:
@        ldrsh.w r4, [pc, #1435]
@      : ldrsh.w r4, [pc, #1435]               @ encoding: [0x3f,0xf9,0x9b,0x45]

@------------------------------------------------------------------------------
@ LDRSHT
@------------------------------------------------------------------------------
        ldrsht r1, [r2]
        ldrsht r1, [r8, #0]
        ldrsht r1, [r8, #3]
        ldrsht r1, [r8, #255]

@ CHECK: ldrsht	r1, [r2]                @ encoding: [0x32,0xf9,0x00,0x1e]
@ CHECK: ldrsht	r1, [r8]                @ encoding: [0x38,0xf9,0x00,0x1e]
@ CHECK: ldrsht	r1, [r8, #3]            @ encoding: [0x38,0xf9,0x03,0x1e]
@ CHECK: ldrsht	r1, [r8, #255]          @ encoding: [0x38,0xf9,0xff,0x1e]


@------------------------------------------------------------------------------
@ LDRT
@------------------------------------------------------------------------------
        ldrt r1, [r2]
        ldrt r2, [r6, #0]
        ldrt r3, [r7, #3]
        ldrt r4, [r9, #255]

@ CHECK: ldrt	r1, [r2]                @ encoding: [0x52,0xf8,0x00,0x1e]
@ CHECK: ldrt	r2, [r6]                @ encoding: [0x56,0xf8,0x00,0x2e]
@ CHECK: ldrt	r3, [r7, #3]            @ encoding: [0x57,0xf8,0x03,0x3e]
@ CHECK: ldrt	r4, [r9, #255]          @ encoding: [0x59,0xf8,0xff,0x4e]


@------------------------------------------------------------------------------
@ LSL (immediate)
@------------------------------------------------------------------------------
        lsl r2, r3, #12
        lsls r8, r3, #31
        lsls.w r2, r3, #1
        lsl r2, r3, #4
        lsls r2, r12, #15

        lsl r3, #19
        lsls r8, #2
        lsls.w r7, #5
        lsl.w r12, #21

        lsls r1, r2, #1
        itt eq
        lslseq r1, r2, #1
        lsleq r1, r2, #1

@ CHECK: lsl.w	r2, r3, #12             @ encoding: [0x4f,0xea,0x03,0x32]
@ CHECK: lsls.w	r8, r3, #31             @ encoding: [0x5f,0xea,0xc3,0x78]
@ CHECK: lsls.w	r2, r3, #1              @ encoding: [0x5f,0xea,0x43,0x02]
@ CHECK: lsl.w	r2, r3, #4              @ encoding: [0x4f,0xea,0x03,0x12]
@ CHECK: lsls.w	r2, r12, #15            @ encoding: [0x5f,0xea,0xcc,0x32]

@ CHECK: lsl.w	r3, r3, #19             @ encoding: [0x4f,0xea,0xc3,0x43]
@ CHECK: lsls.w	r8, r8, #2              @ encoding: [0x5f,0xea,0x88,0x08]
@ CHECK: lsls.w	r7, r7, #5              @ encoding: [0x5f,0xea,0x47,0x17]
@ CHECK: lsl.w	r12, r12, #21           @ encoding: [0x4f,0xea,0x4c,0x5c]

@ CHECK: lsls   r1, r2, #1              @ encoding: [0x51,0x00]
@ CHECK: itt eq                         @ encoding: [0x04,0xbf]
@ CHECK: lslseq.w r1, r2, #1            @ encoding: [0x5f,0xea,0x42,0x01]
@ CHECK: lsleq  r1, r2, #1              @ encoding: [0x51,0x00]

@------------------------------------------------------------------------------
@ LSL (register)
@------------------------------------------------------------------------------
        lsl r3, r4, r2
        lsl.w r1, r2
        lsls r3, r4, r8

@ CHECK: lsl.w	r3, r4, r2              @ encoding: [0x04,0xfa,0x02,0xf3]
@ CHECK: lsl.w	r1, r1, r2              @ encoding: [0x01,0xfa,0x02,0xf1]
@ CHECK: lsls.w	r3, r4, r8              @ encoding: [0x14,0xfa,0x08,0xf3]


@------------------------------------------------------------------------------
@ LSR (immediate)
@------------------------------------------------------------------------------
        lsr r2, r3, #12
        lsrs r8, r3, #32
        lsrs.w r2, r3, #1
        lsr r2, r3, #4
        lsrs r2, r12, #15

        lsr r3, #19
        lsrs r8, #2
        lsrs.w r7, #5
        lsr.w r12, #21

        lsrs  r1, r2, #1
        itt eq
        lsrseq r1, r2, #1
        lsreq r1, r2, #1

@ CHECK: lsr.w	r2, r3, #12             @ encoding: [0x4f,0xea,0x13,0x32]
@ CHECK: lsrs.w	r8, r3, #32             @ encoding: [0x5f,0xea,0x13,0x08]
@ CHECK: lsrs.w	r2, r3, #1              @ encoding: [0x5f,0xea,0x53,0x02]
@ CHECK: lsr.w	r2, r3, #4              @ encoding: [0x4f,0xea,0x13,0x12]
@ CHECK: lsrs.w	r2, r12, #15            @ encoding: [0x5f,0xea,0xdc,0x32]

@ CHECK: lsr.w	r3, r3, #19             @ encoding: [0x4f,0xea,0xd3,0x43]
@ CHECK: lsrs.w	r8, r8, #2              @ encoding: [0x5f,0xea,0x98,0x08]
@ CHECK: lsrs.w	r7, r7, #5              @ encoding: [0x5f,0xea,0x57,0x17]
@ CHECK: lsr.w	r12, r12, #21           @ encoding: [0x4f,0xea,0x5c,0x5c]

@ CHECK: lsrs   r1, r2, #1              @ encoding: [0x51,0x08]
@ CHECK: itt    eq                      @ encoding: [0x04,0xbf]
@ CHECK: lsrseq.w r1, r2, #1            @ encoding: [0x5f,0xea,0x52,0x01]
@ CHECK: lsreq  r1, r2, #1              @ encoding: [0x51,0x08]

@------------------------------------------------------------------------------
@ LSR (register)
@------------------------------------------------------------------------------
        lsr r3, r4, r2
        lsr.w r1, r2
        lsrs r3, r4, r8

@ CHECK: lsr.w	r3, r4, r2              @ encoding: [0x24,0xfa,0x02,0xf3]
@ CHECK: lsr.w	r1, r1, r2              @ encoding: [0x21,0xfa,0x02,0xf1]
@ CHECK: lsrs.w	r3, r4, r8              @ encoding: [0x34,0xfa,0x08,0xf3]

@------------------------------------------------------------------------------
@ MCR/MCR2
@------------------------------------------------------------------------------
        mcr  p7, #1, r5, c1, c1, #4
        mcr2  p7, #1, r5, c1, c1, #4
        mcr p14, #0, r4, c0, c5
        mcr2 p4, #2, r2, c1, c3

@ CHECK: mcr	p7, #1, r5, c1, c1, #4  @ encoding: [0x21,0xee,0x91,0x57]
@ CHECK: mcr2	p7, #1, r5, c1, c1, #4  @ encoding: [0x21,0xfe,0x91,0x57]
@ CHECK: mcr	p14, #0, r4, c0, c5, #0 @ encoding: [0x00,0xee,0x15,0x4e]
@ CHECK: mcr2	p4, #2, r2, c1, c3, #0  @ encoding: [0x41,0xfe,0x13,0x24]


@------------------------------------------------------------------------------
@ MCRR/MCRR2
@------------------------------------------------------------------------------
        mcrr  p7, #15, r5, r4, c1
        mcrr2  p7, #15, r5, r4, c1

@ CHECK: mcrr	p7, #15, r5, r4, c1     @ encoding: [0x44,0xec,0xf1,0x57]
@ CHECK: mcrr2	p7, #15, r5, r4, c1     @ encoding: [0x44,0xfc,0xf1,0x57]


@------------------------------------------------------------------------------
@ MLA/MLS
@------------------------------------------------------------------------------
        mla  r1,r2,r3,r4
        mls  r1,r2,r3,r4

@ CHECK: mla	r1, r2, r3, r4          @ encoding: [0x02,0xfb,0x03,0x41]
@ CHECK: mls	r1, r2, r3, r4          @ encoding: [0x02,0xfb,0x13,0x41]


@------------------------------------------------------------------------------
@ MOV(immediate)
@------------------------------------------------------------------------------
        movs r1, #21
        movs.w r1, #21
        movs r8, #21
        movw r0, #65535
        movw r1, #43777
        movw r1, #43792
        mov.w r0, #0x3fc0000
        mov r0, #0x3fc0000
        movs.w r0, #0x3fc0000
        itte eq
        movseq r1, #12
        moveq r1, #12
        movne.w r1, #12
        mov.w r6, #450
        it lo
        movlo r1, #-1

        @ alias for mvn
        mov r3, #-3
        mov r11, #0xabcd
        movs r0, #1
        it ne
        movne r3, #15

        itt eq
        moveq r0, #255
        moveq r1, #256

@ CHECK: movs	r1, #21                 @ encoding: [0x15,0x21]
@ CHECK: movs.w	r1, #21                 @ encoding: [0x5f,0xf0,0x15,0x01]
@ CHECK: movs.w	r8, #21                 @ encoding: [0x5f,0xf0,0x15,0x08]
@ CHECK: movw	r0, #65535              @ encoding: [0x4f,0xf6,0xff,0x70]
@ CHECK: movw	r1, #43777              @ encoding: [0x4a,0xf6,0x01,0x31]
@ CHECK: movw	r1, #43792              @ encoding: [0x4a,0xf6,0x10,0x31]
@ CHECK: mov.w	r0, #66846720           @ encoding: [0x4f,0xf0,0x7f,0x70]
@ CHECK: mov.w	r0, #66846720           @ encoding: [0x4f,0xf0,0x7f,0x70]
@ CHECK: movs.w	r0, #66846720           @ encoding: [0x5f,0xf0,0x7f,0x70]
@ CHECK: itte	eq                      @ encoding: [0x06,0xbf]
@ CHECK: movseq.w	r1, #12         @ encoding: [0x5f,0xf0,0x0c,0x01]
@ CHECK: moveq	r1, #12                 @ encoding: [0x0c,0x21]
@ CHECK: movne.w r1, #12                @ encoding: [0x4f,0xf0,0x0c,0x01]
@ CHECK: mov.w	r6, #450                @ encoding: [0x4f,0xf4,0xe1,0x76]
@ CHECK: it	lo                      @ encoding: [0x38,0xbf]
@ CHECK: movlo.w	r1, #-1         @ encoding: [0x4f,0xf0,0xff,0x31]
@ CHECK: mvn	r3, #2                  @ encoding: [0x6f,0xf0,0x02,0x03]
@ CHECK: movw	r11, #43981             @ encoding: [0x4a,0xf6,0xcd,0x3b]
@ CHECK: movs	r0, #1                  @ encoding: [0x01,0x20]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: movne	r3, #15                 @ encoding: [0x0f,0x23]

@ CHECK: itt    eq                      @ encoding: [0x04,0xbf]
@ CHECK: moveq  r0, #255                @ encoding: [0xff,0x20]
@ CHECK: movweq r1, #256                @ encoding: [0x40,0xf2,0x00,0x11]

@------------------------------------------------------------------------------
@ MOV(shifted register)
@------------------------------------------------------------------------------
        mov r6, r2, lsl #16
        mov.w r6, r2, lsl #16
        mov r6, r2, lsr #16
        mov.w r6, r2, lsr #16
        movs r6, r2, asr #32
        movs.w r6, r2, asr #32
        movs r6, r2, ror #5
        movs.w r6, r2, ror #5
        movs r4, r4, lsl r5
        movs.w r4, r4, lsl r5
        movs r4, r4, lsr r5
        movs.w r4, r4, lsr r5
        movs r4, r4, asr r5
        movs.w r4, r4, asr r5
        movs r4, r4, ror r5
        movs.w r4, r4, ror r5
        mov r4, r4, lsl r5
        movs r4, r4, ror r8
        movs r4, r5, lsr r6
        itttt eq
        moveq r4, r4, lsl r5
        moveq r4, r4, lsr r5
        moveq r4, r4, asr r5
        moveq r4, r4, ror r5
        mov r4, r4, rrx

@ CHECK: lsl.w	r6, r2, #16             @ encoding: [0x4f,0xea,0x02,0x46]
@ CHECK: lsl.w	r6, r2, #16             @ encoding: [0x4f,0xea,0x02,0x46]
@ CHECK: lsr.w	r6, r2, #16             @ encoding: [0x4f,0xea,0x12,0x46]
@ CHECK: lsr.w	r6, r2, #16             @ encoding: [0x4f,0xea,0x12,0x46]
@ CHECK: asrs	r6, r2, #32             @ encoding: [0x16,0x10]
@ CHECK: asrs.w	r6, r2, #32             @ encoding: [0x5f,0xea,0x22,0x06]
@ CHECK: rors.w	r6, r2, #5              @ encoding: [0x5f,0xea,0x72,0x16]
@ CHECK: rors.w	r6, r2, #5              @ encoding: [0x5f,0xea,0x72,0x16]
@ CHECK: lsls	r4, r5                  @ encoding: [0xac,0x40]
@ CHECK: lsls.w	r4, r4, r5              @ encoding: [0x14,0xfa,0x05,0xf4]
@ CHECK: lsrs	r4, r5                  @ encoding: [0xec,0x40]
@ CHECK: lsrs.w	r4, r4, r5              @ encoding: [0x34,0xfa,0x05,0xf4]
@ CHECK: asrs	r4, r5                  @ encoding: [0x2c,0x41]
@ CHECK: asrs.w	r4, r4, r5              @ encoding: [0x54,0xfa,0x05,0xf4]
@ CHECK: rors	r4, r5                  @ encoding: [0xec,0x41]
@ CHECK: rors.w	r4, r4, r5              @ encoding: [0x74,0xfa,0x05,0xf4]
@ CHECK: lsl.w	r4, r4, r5              @ encoding: [0x04,0xfa,0x05,0xf4]
@ CHECK: rors.w	r4, r4, r8              @ encoding: [0x74,0xfa,0x08,0xf4]
@ CHECK: lsrs.w	r4, r5, r6              @ encoding: [0x35,0xfa,0x06,0xf4]
@ CHECK: itttt	eq                      @ encoding: [0x01,0xbf]
@ CHECK: lsleq	r4, r5                  @ encoding: [0xac,0x40]
@ CHECK: lsreq	r4, r5                  @ encoding: [0xec,0x40]
@ CHECK: asreq	r4, r5                  @ encoding: [0x2c,0x41]
@ CHECK: roreq	r4, r5                  @ encoding: [0xec,0x41]
@ CHECK: rrx	r4, r4                  @ encoding: [0x4f,0xea,0x34,0x04]


@------------------------------------------------------------------------------
@ MOVT
@------------------------------------------------------------------------------
        movt r3, #7
        movt r6, #0xffff
        it eq
        movteq r4, #0xff0

@ CHECK: movt	r3, #7                  @ encoding: [0xc0,0xf2,0x07,0x03]
@ CHECK: movt	r6, #65535              @ encoding: [0xcf,0xf6,0xff,0x76]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: movteq	r4, #4080               @ encoding: [0xc0,0xf6,0xf0,0x74]

@------------------------------------------------------------------------------
@ MRC/MRC2
@------------------------------------------------------------------------------
        mrc  p14, #0, r1, c1, c2, #4
        mrc  p15, #7, apsr_nzcv, c15, c6, #6
        mrc  p9, #1, r1, c2, c2
        mrc2 p12, #3, r3, c3, c4
        mrc2 p14, #0, r1, c1, c2, #4
        mrc2 p8, #7, apsr_nzcv, c15, c0, #1
 
@ CHECK: mrc  p14, #0, r1, c1, c2, #4            @ encoding: [0x11,0xee,0x92,0x1e]
@ CHECK: mrc  p15, #7, apsr_nzcv, c15, c6, #6    @ encoding: [0xff,0xee,0xd6,0xff]
@ CHECK: mrc  p9, #1, r1, c2, c2, #0             @ encoding: [0x32,0xee,0x12,0x19]
@ CHECK: mrc2 p12, #3, r3, c3, c4, #0            @ encoding: [0x73,0xfe,0x14,0x3c]
@ CHECK: mrc2 p14, #0, r1, c1, c2, #4            @ encoding: [0x11,0xfe,0x92,0x1e]
@ CHECK: mrc2 p8, #7, apsr_nzcv, c15, c0, #1     @ encoding: [0xff,0xfe,0x30,0xf8]
 
@------------------------------------------------------------------------------
@ MRRC/MRRC2
@------------------------------------------------------------------------------
        mrrc  p7, #1, r5, r4, c1
        mrrc2  p7, #1, r5, r4, c1

@ CHECK: mrrc	p7, #1, r5, r4, c1      @ encoding: [0x54,0xec,0x11,0x57]
@ CHECK: mrrc2	p7, #1, r5, r4, c1      @ encoding: [0x54,0xfc,0x11,0x57]


@------------------------------------------------------------------------------
@ MRS
@------------------------------------------------------------------------------
        mrs  r8, apsr
        mrs  r8, cpsr
        mrs  r8, spsr

@ CHECK: mrs	r8, apsr                @ encoding: [0xef,0xf3,0x00,0x88]
@ CHECK: mrs	r8, apsr                @ encoding: [0xef,0xf3,0x00,0x88]
@ CHECK: mrs	r8, spsr                @ encoding: [0xff,0xf3,0x00,0x88]


@------------------------------------------------------------------------------
@ MSR
@------------------------------------------------------------------------------
        msr  apsr, r1
        msr  apsr_g, r2
        msr  apsr_nzcvq, r3
        msr  APSR_nzcvq, r4
        msr  apsr_nzcvqg, r5
        msr  cpsr_fc, r6
        msr  cpsr_c, r7
        msr  cpsr_x, r8
        msr  cpsr_fc, r9
        msr  cpsr_all, r11
        msr  cpsr_fsx, r12
        msr  spsr_fc, r0
        msr  SPSR_fsxc, r5
        msr  cpsr_fsxc, r8
        msr  cpsr, r3

@ CHECK: msr	APSR_nzcvq, r1          @ encoding: [0x81,0xf3,0x00,0x88]
@ CHECK: msr	APSR_g, r2              @ encoding: [0x82,0xf3,0x00,0x84]
@ CHECK: msr	APSR_nzcvq, r3          @ encoding: [0x83,0xf3,0x00,0x88]
@ CHECK: msr	APSR_nzcvq, r4          @ encoding: [0x84,0xf3,0x00,0x88]
@ CHECK: msr	APSR_nzcvqg, r5         @ encoding: [0x85,0xf3,0x00,0x8c]
@ CHECK: msr	CPSR_fc, r6             @ encoding: [0x86,0xf3,0x00,0x89]
@ CHECK: msr	CPSR_c, r7              @ encoding: [0x87,0xf3,0x00,0x81]
@ CHECK: msr	CPSR_x, r8              @ encoding: [0x88,0xf3,0x00,0x82]
@ CHECK: msr	CPSR_fc, r9             @ encoding: [0x89,0xf3,0x00,0x89]
@ CHECK: msr	CPSR_fc, r11            @ encoding: [0x8b,0xf3,0x00,0x89]
@ CHECK: msr	CPSR_fsx, r12           @ encoding: [0x8c,0xf3,0x00,0x8e]
@ CHECK: msr	SPSR_fc, r0             @ encoding: [0x90,0xf3,0x00,0x89]
@ CHECK: msr	SPSR_fsxc, r5           @ encoding: [0x95,0xf3,0x00,0x8f]
@ CHECK: msr	CPSR_fsxc, r8           @ encoding: [0x88,0xf3,0x00,0x8f]
@ CHECK: msr	CPSR_fc, r3             @ encoding: [0x83,0xf3,0x00,0x89]


@------------------------------------------------------------------------------
@ MUL
@------------------------------------------------------------------------------
        muls r3, r4, r3
        mul r3, r4, r3
        mul r3, r4, r6
        it eq
        muleq r3, r4, r5
        it le
        mulle r4, r4, r8
        mul r5, r6

@ CHECK: muls	r3, r4, r3              @ encoding: [0x63,0x43]
@ CHECK: mul	r3, r4, r3              @ encoding: [0x04,0xfb,0x03,0xf3]
@ CHECK: mul	r3, r4, r6              @ encoding: [0x04,0xfb,0x06,0xf3]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: muleq	r3, r4, r5              @ encoding: [0x04,0xfb,0x05,0xf3]
@ CHECK: it	le                      @ encoding: [0xd8,0xbf]
@ CHECK: mulle	r4, r4, r8              @ encoding: [0x04,0xfb,0x08,0xf4]
@ CHECK: mul	r5, r6, r5              @ encoding: [0x06,0xfb,0x05,0xf5]


@------------------------------------------------------------------------------
@ MVN(immediate)
@------------------------------------------------------------------------------
        mvns r8, #21
        mvn r0, #0x3fc0000
        mvns r0, #0x3fc0000
        itte eq
        mvnseq r1, #12
        mvneq.w r1, #12
        mvnne r1, #12

@ CHECK: mvns	r8, #21                 @ encoding: [0x7f,0xf0,0x15,0x08]
@ CHECK: mvn	r0, #66846720           @ encoding: [0x6f,0xf0,0x7f,0x70]
@ CHECK: mvns	r0, #66846720           @ encoding: [0x7f,0xf0,0x7f,0x70]
@ CHECK: itte	eq                      @ encoding: [0x06,0xbf]
@ CHECK: mvnseq	r1, #12                 @ encoding: [0x7f,0xf0,0x0c,0x01]
@ CHECK: mvneq	r1, #12                 @ encoding: [0x6f,0xf0,0x0c,0x01]
@ CHECK: mvnne	r1, #12                 @ encoding: [0x6f,0xf0,0x0c,0x01]


@------------------------------------------------------------------------------
@ MVN(register)
@------------------------------------------------------------------------------
        mvn r2, r3
        mvns r2, r3
        mvn r5, r6, lsl #19
        mvn r5, r6, lsr #9
        mvn.w r5, r6, asr #4
        mvn r5, r6, ror #6
        mvn r5, r6, rrx
        it eq
        mvneq r2, r3

@ CHECK: mvn.w	r2, r3                  @ encoding: [0x6f,0xea,0x03,0x02]
@ CHECK: mvns	r2, r3                  @ encoding: [0xda,0x43]
@ CHECK: mvn.w	r5, r6, lsl #19         @ encoding: [0x6f,0xea,0xc6,0x45]
@ CHECK: mvn.w	r5, r6, lsr #9          @ encoding: [0x6f,0xea,0x56,0x25]
@ CHECK: mvn.w	r5, r6, asr #4          @ encoding: [0x6f,0xea,0x26,0x15]
@ CHECK: mvn.w	r5, r6, ror #6          @ encoding: [0x6f,0xea,0xb6,0x15]
@ CHECK: mvn.w	r5, r6, rrx             @ encoding: [0x6f,0xea,0x36,0x05]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: mvneq	r2, r3                  @ encoding: [0xda,0x43]

@------------------------------------------------------------------------------
@ NEG
@------------------------------------------------------------------------------
        neg r5, r2
        neg r5, r8

@ CHECK: rsb.w	r5, r2, #0              @ encoding: [0xc2,0xf1,0x00,0x05]
@ CHECK: rsb.w	r5, r8, #0              @ encoding: [0xc8,0xf1,0x00,0x05]


@------------------------------------------------------------------------------
@ NOP
@------------------------------------------------------------------------------
        nop.w

@ CHECK: nop.w                          @ encoding: [0xaf,0xf3,0x00,0x80]


@------------------------------------------------------------------------------
@ ORN
@------------------------------------------------------------------------------
        orn r4, r5, #0xf000
        orn r4, r5, r6
        orns r4, r5, r6
        orn r4, r5, r6, lsl #5
        orns r4, r5, r6, lsr #5
        orn r4, r5, r6, lsr #5
        orns r4, r5, r6, asr #5
        orn r4, r5, r6, ror #5

@ CHECK: orn	r4, r5, #61440          @ encoding: [0x65,0xf4,0x70,0x44]
@ CHECK: orn	r4, r5, r6              @ encoding: [0x65,0xea,0x06,0x04]
@ CHECK: orns	r4, r5, r6              @ encoding: [0x75,0xea,0x06,0x04]
@ CHECK: orn	r4, r5, r6, lsl #5      @ encoding: [0x65,0xea,0x46,0x14]
@ CHECK: orns	r4, r5, r6, lsr #5      @ encoding: [0x75,0xea,0x56,0x14]
@ CHECK: orn	r4, r5, r6, lsr #5      @ encoding: [0x65,0xea,0x56,0x14]
@ CHECK: orns	r4, r5, r6, asr #5      @ encoding: [0x75,0xea,0x66,0x14]
@ CHECK: orn	r4, r5, r6, ror #5      @ encoding: [0x65,0xea,0x76,0x14]


@------------------------------------------------------------------------------
@ ORR
@------------------------------------------------------------------------------
        orr r4, r5, #0xf000
        orr r4, r5, r6
        orr r4, r5, r6, lsl #5
        orrs r4, r5, r6, lsr #5
        orr r4, r5, r6, lsr #5
        orrs r4, r5, r6, asr #5
        orr r4, r5, r6, ror #5

@ CHECK: orr	r4, r5, #61440          @ encoding: [0x45,0xf4,0x70,0x44]
@ CHECK: orr.w	r4, r5, r6              @ encoding: [0x45,0xea,0x06,0x04]
@ CHECK: orr.w	r4, r5, r6, lsl #5      @ encoding: [0x45,0xea,0x46,0x14]
@ CHECK: orrs.w	r4, r5, r6, lsr #5      @ encoding: [0x55,0xea,0x56,0x14]
@ CHECK: orr.w	r4, r5, r6, lsr #5      @ encoding: [0x45,0xea,0x56,0x14]
@ CHECK: orrs.w	r4, r5, r6, asr #5      @ encoding: [0x55,0xea,0x66,0x14]
@ CHECK: orr.w	r4, r5, r6, ror #5      @ encoding: [0x45,0xea,0x76,0x14]


@------------------------------------------------------------------------------
@ PKH
@------------------------------------------------------------------------------
        pkhbt r2, r2, r3
        pkhbt r2, r2, r3, lsl #31
        pkhbt r2, r2, r3, lsl #0
        pkhbt r2, r2, r3, lsl #15

        pkhtb r2, r2, r3
        pkhtb r2, r2, r3, asr #31
        pkhtb r2, r2, r3, asr #15

@ CHECK: pkhbt	r2, r2, r3              @ encoding: [0xc2,0xea,0x03,0x02]
@ CHECK: pkhbt	r2, r2, r3, lsl #31     @ encoding: [0xc2,0xea,0xc3,0x72]
@ CHECK: pkhbt	r2, r2, r3              @ encoding: [0xc2,0xea,0x03,0x02]
@ CHECK: pkhbt	r2, r2, r3, lsl #15     @ encoding: [0xc2,0xea,0xc3,0x32]

@ CHECK: pkhbt	r2, r3, r2              @ encoding: [0xc3,0xea,0x02,0x02]
@ CHECK: pkhtb	r2, r2, r3, asr #31     @ encoding: [0xc2,0xea,0xe3,0x72]
@ CHECK: pkhtb	r2, r2, r3, asr #15     @ encoding: [0xc2,0xea,0xe3,0x32]


@------------------------------------------------------------------------------
@ PLD(immediate)
@------------------------------------------------------------------------------
        pld [r5, #-4]
        pld [r6, #32]
        pld [r6, #33]
        pld [r6, #257]
        pld [r7, #257]
        pld [r1, #0]
        pld [r1, #-0]

@ CHECK: pld	[r5, #-4]               @ encoding: [0x15,0xf8,0x04,0xfc]
@ CHECK: pld	[r6, #32]               @ encoding: [0x96,0xf8,0x20,0xf0]
@ CHECK: pld	[r6, #33]               @ encoding: [0x96,0xf8,0x21,0xf0]
@ CHECK: pld	[r6, #257]              @ encoding: [0x96,0xf8,0x01,0xf1]
@ CHECK: pld	[r7, #257]              @ encoding: [0x97,0xf8,0x01,0xf1]
@ CHECK: pld	[r1]                    @ encoding: [0x91,0xf8,0x00,0xf0]
@ CHECK: pld	[r1, #-0]               @ encoding: [0x11,0xf8,0x00,0xfc]


@------------------------------------------------------------------------------
@ PLD(literal)
@------------------------------------------------------------------------------
@        pld  _foo

@ FIXME: pld	_foo                    @ encoding: [0x9f'A',0xf8'A',A,0xf0'A']
            @   fixup A - offset: 0, value: _foo, kind: fixup_t2_ldst_pcrel_12

        pld [pc,#-4095]
@ CHECK: pld [pc, #-4095]            @ encoding: [0x1f,0xf8,0xff,0xff]


@------------------------------------------------------------------------------
@ PLD(register)
@------------------------------------------------------------------------------
        pld [r8, r1]
        pld [r5, r2]
        pld [r0, r2, lsl #3]
        pld [r8, r2, lsl #2]
        pld [sp, r2, lsl #1]
        pld [sp, r2, lsl #0]

@ CHECK: pld	[r8, r1]                @ encoding: [0x18,0xf8,0x01,0xf0]
@ CHECK: pld	[r5, r2]                @ encoding: [0x15,0xf8,0x02,0xf0]
@ CHECK: pld	[r0, r2, lsl #3]        @ encoding: [0x10,0xf8,0x32,0xf0]
@ CHECK: pld	[r8, r2, lsl #2]        @ encoding: [0x18,0xf8,0x22,0xf0]
@ CHECK: pld	[sp, r2, lsl #1]        @ encoding: [0x1d,0xf8,0x12,0xf0]
@ CHECK: pld	[sp, r2]                @ encoding: [0x1d,0xf8,0x02,0xf0]

@------------------------------------------------------------------------------
@ PLI(immediate)
@------------------------------------------------------------------------------
        pli [r5, #-4]
        pli [r6, #32]
        pli [r6, #33]
        pli [r6, #257]
        pli [r7, #257]
        pli [pc, #+4095]
        pli [pc, #-4095]

@ CHECK: pli	[r5, #-4]               @ encoding: [0x15,0xf9,0x04,0xfc]
@ CHECK: pli	[r6, #32]               @ encoding: [0x96,0xf9,0x20,0xf0]
@ CHECK: pli	[r6, #33]               @ encoding: [0x96,0xf9,0x21,0xf0]
@ CHECK: pli	[r6, #257]              @ encoding: [0x96,0xf9,0x01,0xf1]
@ CHECK: pli	[r7, #257]              @ encoding: [0x97,0xf9,0x01,0xf1]
@ CHECK: pli    [pc, #4095]             @ encoding: [0x9f,0xf9,0xff,0xff]
@ CHECK: pli    [pc, #-4095]            @ encoding: [0x1f,0xf9,0xff,0xff]


@------------------------------------------------------------------------------
@ PLI(literal)
@------------------------------------------------------------------------------
@        pli  _foo


@ FIXME: pli	_foo                    @ encoding: [0x9f'A',0xf9'A',A,0xf0'A']
           @   fixup A - offset: 0, value: _foo, kind: fixup_t2_ldst_pcrel_12


@------------------------------------------------------------------------------
@ PLI(register)
@------------------------------------------------------------------------------
        pli [r8, r1]
        pli [r5, r2]
        pli [r0, r2, lsl #3]
        pli [r8, r2, lsl #2]
        pli [sp, r2, lsl #1]
        pli [sp, r2, lsl #0]

@ CHECK: pli	[r8, r1]                @ encoding: [0x18,0xf9,0x01,0xf0]
@ CHECK: pli	[r5, r2]                @ encoding: [0x15,0xf9,0x02,0xf0]
@ CHECK: pli	[r0, r2, lsl #3]        @ encoding: [0x10,0xf9,0x32,0xf0]
@ CHECK: pli	[r8, r2, lsl #2]        @ encoding: [0x18,0xf9,0x22,0xf0]
@ CHECK: pli	[sp, r2, lsl #1]        @ encoding: [0x1d,0xf9,0x12,0xf0]
@ CHECK: pli	[sp, r2]                @ encoding: [0x1d,0xf9,0x02,0xf0]

@------------------------------------------------------------------------------
@ POP (alias)
@------------------------------------------------------------------------------
        pop {r2, r9}

@ CHECK: pop.w	{r2, r9}                @ encoding: [0xbd,0xe8,0x04,0x02]


@------------------------------------------------------------------------------
@ PUSH (alias)
@------------------------------------------------------------------------------
        push {r2, r9}

@ CHECK: push.w	{r2, r9}                @ encoding: [0x2d,0xe9,0x04,0x02]


@------------------------------------------------------------------------------
@ QADD/QADD16/QADD8
@------------------------------------------------------------------------------
        qadd r1, r2, r3
        qadd16 r1, r2, r3
        qadd8 r1, r2, r3
        itte gt
        qaddgt r1, r2, r3
        qadd16gt r1, r2, r3
        qadd8le r1, r2, r3

@ CHECK: qadd	r1, r2, r3              @ encoding: [0x83,0xfa,0x82,0xf1]
@ CHECK: qadd16	r1, r2, r3              @ encoding: [0x92,0xfa,0x13,0xf1]
@ CHECK: qadd8	r1, r2, r3              @ encoding: [0x82,0xfa,0x13,0xf1]
@ CHECK: itte	gt                      @ encoding: [0xc6,0xbf]
@ CHECK: qaddgt	r1, r2, r3              @ encoding: [0x83,0xfa,0x82,0xf1]
@ CHECK: qadd16gt r1, r2, r3            @ encoding: [0x92,0xfa,0x13,0xf1]
@ CHECK: qadd8le r1, r2, r3             @ encoding: [0x82,0xfa,0x13,0xf1]


@------------------------------------------------------------------------------
@ QDADD/QDSUB
@------------------------------------------------------------------------------
        qdadd r6, r7, r8
        qdsub r6, r7, r8
        itt hi
        qdaddhi r6, r7, r8
        qdsubhi r6, r7, r8

@ CHECK: qdadd	r6, r7, r8              @ encoding: [0x88,0xfa,0x97,0xf6]
@ CHECK: qdsub	r6, r7, r8              @ encoding: [0x88,0xfa,0xb7,0xf6]
@ CHECK: itt	hi                      @ encoding: [0x84,0xbf]
@ CHECK: qdaddhi r6, r7, r8             @ encoding: [0x88,0xfa,0x97,0xf6]
@ CHECK: qdsubhi r6, r7, r8             @ encoding: [0x88,0xfa,0xb7,0xf6]


@------------------------------------------------------------------------------
@ QSAX
@------------------------------------------------------------------------------
        qsax r9, r12, r0
        it eq
        qsaxeq r9, r12, r0

@ CHECK: qsax	r9, r12, r0             @ encoding: [0xec,0xfa,0x10,0xf9]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: qsaxeq	r9, r12, r0             @ encoding: [0xec,0xfa,0x10,0xf9]


@------------------------------------------------------------------------------
@ QSUB/QSUB16/QSUB8
@------------------------------------------------------------------------------
        qsub r1, r2, r3
        qsub16 r1, r2, r3
        qsub8 r1, r2, r3
        itet le
        qsuble r1, r2, r3
        qsub16gt r1, r2, r3
        qsub8le r1, r2, r3

@ CHECK: qsub	r1, r2, r3              @ encoding: [0x83,0xfa,0xa2,0xf1]
@ CHECK: qsub16	r1, r2, r3              @ encoding: [0xd2,0xfa,0x13,0xf1]
@ CHECK: qsub8	r1, r2, r3              @ encoding: [0xc2,0xfa,0x13,0xf1]
@ CHECK: itet	le                      @ encoding: [0xd6,0xbf]
@ CHECK: qsuble	r1, r2, r3              @ encoding: [0x83,0xfa,0xa2,0xf1]
@ CHECK: qsub16gt	r1, r2, r3      @ encoding: [0xd2,0xfa,0x13,0xf1]
@ CHECK: qsub8le r1, r2, r3             @ encoding: [0xc2,0xfa,0x13,0xf1]


@------------------------------------------------------------------------------
@ RBIT
@------------------------------------------------------------------------------
        rbit r1, r2
        it ne
        rbitne r1, r2

@ CHECK: rbit	r1, r2                  @ encoding: [0x92,0xfa,0xa2,0xf1]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: rbitne	r1, r2                  @ encoding: [0x92,0xfa,0xa2,0xf1]


@------------------------------------------------------------------------------
@ REV
@------------------------------------------------------------------------------
        rev.w r1, r2
        rev r2, r8
        itt ne
        revne r1, r2
        revne r1, r8

@ CHECK: rev.w	r1, r2                  @ encoding: [0x92,0xfa,0x82,0xf1]
@ CHECK: rev.w	r2, r8                  @ encoding: [0x98,0xfa,0x88,0xf2]
@ CHECK: itt	ne                      @ encoding: [0x1c,0xbf]
@ CHECK: revne	r1, r2                  @ encoding: [0x11,0xba]
@ CHECK: revne.w r1, r8                 @ encoding: [0x98,0xfa,0x88,0xf1]


@------------------------------------------------------------------------------
@ REV16
@------------------------------------------------------------------------------
        rev16.w r1, r2
        rev16 r2, r8
        itt ne
        rev16ne r1, r2
        rev16ne r1, r8

@ CHECK: rev16.w r1, r2                 @ encoding: [0x92,0xfa,0x92,0xf1]
@ CHECK: rev16.w r2, r8                 @ encoding: [0x98,0xfa,0x98,0xf2]
@ CHECK: itt	ne                      @ encoding: [0x1c,0xbf]
@ CHECK: rev16ne r1, r2                 @ encoding: [0x51,0xba]
@ CHECK: rev16ne.w	r1, r8          @ encoding: [0x98,0xfa,0x98,0xf1]


@------------------------------------------------------------------------------
@ REVSH
@------------------------------------------------------------------------------
        revsh.w r1, r2
        revsh r2, r8
        itt ne
        revshne r1, r2
        revshne r1, r8

@ CHECK: revsh.w r1, r2                 @ encoding: [0x92,0xfa,0xb2,0xf1]
@ CHECK: revsh.w r2, r8                 @ encoding: [0x98,0xfa,0xb8,0xf2]
@ CHECK: itt	ne                      @ encoding: [0x1c,0xbf]
@ CHECK: revshne r1, r2                 @ encoding: [0xd1,0xba]
@ CHECK: revshne.w	r1, r8          @ encoding: [0x98,0xfa,0xb8,0xf1]


@------------------------------------------------------------------------------
@ ROR (immediate)
@------------------------------------------------------------------------------
        ror r2, r3, #12
        rors r8, r3, #31
        rors.w r2, r3, #1
        ror r2, r3, #4
        rors r2, r12, #15

        ror r3, #19
        rors r8, #2
        rors.w r7, #5
        ror.w r12, #21

@ CHECK: ror.w	r2, r3, #12             @ encoding: [0x4f,0xea,0x33,0x32]
@ CHECK: rors.w	r8, r3, #31             @ encoding: [0x5f,0xea,0xf3,0x78]
@ CHECK: rors.w	r2, r3, #1              @ encoding: [0x5f,0xea,0x73,0x02]
@ CHECK: ror.w	r2, r3, #4              @ encoding: [0x4f,0xea,0x33,0x12]
@ CHECK: rors.w	r2, r12, #15            @ encoding: [0x5f,0xea,0xfc,0x32]

@ CHECK: ror.w	r3, r3, #19             @ encoding: [0x4f,0xea,0xf3,0x43]
@ CHECK: rors.w	r8, r8, #2              @ encoding: [0x5f,0xea,0xb8,0x08]
@ CHECK: rors.w	r7, r7, #5              @ encoding: [0x5f,0xea,0x77,0x17]
@ CHECK: ror.w	r12, r12, #21           @ encoding: [0x4f,0xea,0x7c,0x5c]


@------------------------------------------------------------------------------
@ ROR (register)
@------------------------------------------------------------------------------
        ror r3, r4, r2
        ror.w r1, r2
        rors r3, r4, r8

@ CHECK: ror.w	r3, r4, r2              @ encoding: [0x64,0xfa,0x02,0xf3]
@ CHECK: ror.w	r1, r1, r2              @ encoding: [0x61,0xfa,0x02,0xf1]
@ CHECK: rors.w	r3, r4, r8              @ encoding: [0x74,0xfa,0x08,0xf3]


@------------------------------------------------------------------------------
@ RRX
@------------------------------------------------------------------------------
        rrx r1, r2
        rrxs r1, r2
        ite lt
        rrxlt r9, r12
        rrxsge r8, r3

@ CHECK: rrx	r1, r2                  @ encoding: [0x4f,0xea,0x32,0x01]
@ CHECK: rrxs	r1, r2                  @ encoding: [0x5f,0xea,0x32,0x01]
@ CHECK: ite	lt                      @ encoding: [0xb4,0xbf]
@ CHECK: rrxlt	r9, r12                 @ encoding: [0x4f,0xea,0x3c,0x09]
@ CHECK: rrxsge	r8, r3                  @ encoding: [0x5f,0xea,0x33,0x08]

@------------------------------------------------------------------------------
@ RSB (immediate)
@------------------------------------------------------------------------------
        rsb r2, r5, #0xff000
        rsbs r3, r12, #0xf
        rsb r1, #0xff
        rsb r1, r1, #0xff
        rsb r11, r11, #0
        rsb r9, #0
        rsbs r3, r1, #0
        rsb r3, r1, #0

@ CHECK: rsb.w	r2, r5, #1044480        @ encoding: [0xc5,0xf5,0x7f,0x22]
@ CHECK: rsbs.w	r3, r12, #15            @ encoding: [0xdc,0xf1,0x0f,0x03]
@ CHECK: rsb.w	r1, r1, #255            @ encoding: [0xc1,0xf1,0xff,0x01]
@ CHECK: rsb.w	r1, r1, #255            @ encoding: [0xc1,0xf1,0xff,0x01]
@ CHECK: rsb.w	r11, r11, #0            @ encoding: [0xcb,0xf1,0x00,0x0b]
@ CHECK: rsb.w	r9, r9, #0              @ encoding: [0xc9,0xf1,0x00,0x09]
@ CHECK: rsbs	r3, r1, #0              @ encoding: [0x4b,0x42]
@ CHECK: rsb.w	r3, r1, #0              @ encoding: [0xc1,0xf1,0x00,0x03]


@------------------------------------------------------------------------------
@ RSB (register)
@------------------------------------------------------------------------------
        rsb r4, r8
        rsb r4, r9, r8
        rsb r1, r4, r8, asr #3
        rsbs r2, r1, r7, lsl #1

@ CHECK: rsb	r4, r4, r8              @ encoding: [0xc4,0xeb,0x08,0x04]
@ CHECK: rsb	r4, r9, r8              @ encoding: [0xc9,0xeb,0x08,0x04]
@ CHECK: rsb	r1, r4, r8, asr #3      @ encoding: [0xc4,0xeb,0xe8,0x01]
@ CHECK: rsbs	r2, r1, r7, lsl #1      @ encoding: [0xd1,0xeb,0x47,0x02]


@------------------------------------------------------------------------------
@ SADD16
@------------------------------------------------------------------------------
        sadd16 r3, r4, r8
        it ne
        sadd16ne r3, r4, r8

@ CHECK: sadd16	r3, r4, r8              @ encoding: [0x94,0xfa,0x08,0xf3]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: sadd16ne	r3, r4, r8      @ encoding: [0x94,0xfa,0x08,0xf3]


@------------------------------------------------------------------------------
@ SADD8
@------------------------------------------------------------------------------
        sadd8 r3, r4, r8
        it ne
        sadd8ne r3, r4, r8

@ CHECK: sadd8	r3, r4, r8              @ encoding: [0x84,0xfa,0x08,0xf3]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: sadd8ne r3, r4, r8             @ encoding: [0x84,0xfa,0x08,0xf3]


@------------------------------------------------------------------------------
@ SASX
@------------------------------------------------------------------------------
        saddsubx r9, r2, r7
        it ne
        saddsubxne r2, r5, r6
        sasx r9, r2, r7
        it ne
        sasxne r2, r5, r6

@ CHECK: sasx	r9, r2, r7              @ encoding: [0xa2,0xfa,0x07,0xf9]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: sasxne	r2, r5, r6              @ encoding: [0xa5,0xfa,0x06,0xf2]
@ CHECK: sasx	r9, r2, r7              @ encoding: [0xa2,0xfa,0x07,0xf9]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: sasxne	r2, r5, r6              @ encoding: [0xa5,0xfa,0x06,0xf2]


@------------------------------------------------------------------------------
@ SBC (immediate)
@------------------------------------------------------------------------------
        sbc r0, r1, #4
        sbcs r0, r1, #0
        sbc r1, r2, #255
        sbc r3, r7, #0x00550055
        sbc r8, r12, #0xaa00aa00
        sbc r9, r7, #0xa5a5a5a5
        sbc r5, r3, #0x87000000
        sbc r4, r2, #0x7f800000
        sbc r4, r2, #0x00000680

@ CHECK: sbc	r0, r1, #4              @ encoding: [0x61,0xf1,0x04,0x00]
@ CHECK: sbcs	r0, r1, #0              @ encoding: [0x71,0xf1,0x00,0x00]
@ CHECK: sbc	r1, r2, #255            @ encoding: [0x62,0xf1,0xff,0x01]
@ CHECK: sbc	r3, r7, #5570645        @ encoding: [0x67,0xf1,0x55,0x13]
@ CHECK: sbc	r8, r12, #2852170240    @ encoding: [0x6c,0xf1,0xaa,0x28]
@ CHECK: sbc	r9, r7, #2779096485     @ encoding: [0x67,0xf1,0xa5,0x39]
@ CHECK: sbc	r5, r3, #2264924160     @ encoding: [0x63,0xf1,0x07,0x45]
@ CHECK: sbc	r4, r2, #2139095040     @ encoding: [0x62,0xf1,0xff,0x44]
@ CHECK: sbc	r4, r2, #1664           @ encoding: [0x62,0xf5,0xd0,0x64]


@------------------------------------------------------------------------------
@ SBC (register)
@------------------------------------------------------------------------------
        sbc r4, r5, r6
        sbcs r4, r5, r6
        sbc.w r9, r1, r3
        sbcs.w r9, r1, r3
        sbc	r0, r1, r3, ror #4
        sbcs	r0, r1, r3, lsl #7
        sbc.w	r0, r1, r3, lsr #31
        sbcs.w	r0, r1, r3, asr #32

@ CHECK: sbc.w	r4, r5, r6              @ encoding: [0x65,0xeb,0x06,0x04]
@ CHECK: sbcs.w	r4, r5, r6              @ encoding: [0x75,0xeb,0x06,0x04]
@ CHECK: sbc.w	r9, r1, r3              @ encoding: [0x61,0xeb,0x03,0x09]
@ CHECK: sbcs.w	r9, r1, r3              @ encoding: [0x71,0xeb,0x03,0x09]
@ CHECK: sbc.w	r0, r1, r3, ror #4      @ encoding: [0x61,0xeb,0x33,0x10]
@ CHECK: sbcs.w	r0, r1, r3, lsl #7      @ encoding: [0x71,0xeb,0xc3,0x10]
@ CHECK: sbc.w	r0, r1, r3, lsr #31     @ encoding: [0x61,0xeb,0xd3,0x70]
@ CHECK: sbcs.w	r0, r1, r3, asr #32     @ encoding: [0x71,0xeb,0x23,0x00]


@------------------------------------------------------------------------------
@ SBFX
@------------------------------------------------------------------------------
        sbfx r4, r5, #16, #1
        it gt
        sbfxgt r4, r5, #16, #16

@ CHECK: sbfx	r4, r5, #16, #1         @ encoding: [0x45,0xf3,0x00,0x44]
@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: sbfxgt	r4, r5, #16, #16        @ encoding: [0x45,0xf3,0x0f,0x44]


@------------------------------------------------------------------------------
@ SEL
@------------------------------------------------------------------------------
        sel r5, r9, r2
        it le
        selle r5, r9, r2

@ CHECK: sel	r5, r9, r2              @ encoding: [0xa9,0xfa,0x82,0xf5]
@ CHECK: it	le                      @ encoding: [0xd8,0xbf]
@ CHECK: selle	r5, r9, r2              @ encoding: [0xa9,0xfa,0x82,0xf5]


@------------------------------------------------------------------------------
@ SEV
@------------------------------------------------------------------------------
        sev.w
        it eq
        seveq.w

@ CHECK: sev.w                           @ encoding: [0xaf,0xf3,0x04,0x80]
@ CHECK: it	eq                       @ encoding: [0x08,0xbf]
@ CHECK: seveq.w                         @ encoding: [0xaf,0xf3,0x04,0x80]


@------------------------------------------------------------------------------
@ SADD16/SADD8
@------------------------------------------------------------------------------
        sadd16 r1, r2, r3
        sadd8 r1, r2, r3
        ite gt
        sadd16gt r1, r2, r3
        sadd8le r1, r2, r3

@ CHECK: sadd16	r1, r2, r3              @ encoding: [0x92,0xfa,0x03,0xf1]
@ CHECK: sadd8	r1, r2, r3              @ encoding: [0x82,0xfa,0x03,0xf1]
@ CHECK: ite	gt                      @ encoding: [0xcc,0xbf]
@ CHECK: sadd16gt	r1, r2, r3      @ encoding: [0x92,0xfa,0x03,0xf1]
@ CHECK: sadd8le r1, r2, r3             @ encoding: [0x82,0xfa,0x03,0xf1]


@------------------------------------------------------------------------------
@ SHASX
@------------------------------------------------------------------------------
        shasx r4, r8, r2
        it gt
        shasxgt r4, r8, r2
        shaddsubx r4, r8, r2
        it gt
        shaddsubxgt r4, r8, r2

@ CHECK: shasx	r4, r8, r2              @ encoding: [0xa8,0xfa,0x22,0xf4]
@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: shasxgt r4, r8, r2             @ encoding: [0xa8,0xfa,0x22,0xf4]
@ CHECK: shasx	r4, r8, r2              @ encoding: [0xa8,0xfa,0x22,0xf4]
@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: shasxgt r4, r8, r2             @ encoding: [0xa8,0xfa,0x22,0xf4]


@------------------------------------------------------------------------------
@ SHASX
@------------------------------------------------------------------------------
        shsax r4, r8, r2
        it gt
        shsaxgt r4, r8, r2
        shsubaddx r4, r8, r2
        it gt
        shsubaddxgt r4, r8, r2

@ CHECK: shsax	r4, r8, r2              @ encoding: [0xe8,0xfa,0x22,0xf4]
@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: shsaxgt r4, r8, r2             @ encoding: [0xe8,0xfa,0x22,0xf4]
@ CHECK: shsax	r4, r8, r2              @ encoding: [0xe8,0xfa,0x22,0xf4]
@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: shsaxgt r4, r8, r2             @ encoding: [0xe8,0xfa,0x22,0xf4]


@------------------------------------------------------------------------------
@ SHSUB16/SHSUB8
@------------------------------------------------------------------------------
        shsub16 r4, r8, r2
        shsub8 r4, r8, r2
        itt gt
        shsub16gt r4, r8, r2
        shsub8gt r4, r8, r2

@ CHECK: shsub16 r4, r8, r2             @ encoding: [0xd8,0xfa,0x22,0xf4]
@ CHECK: shsub8	r4, r8, r2              @ encoding: [0xc8,0xfa,0x22,0xf4]
@ CHECK: itt	gt                      @ encoding: [0xc4,0xbf]
@ CHECK: shsub16gt	r4, r8, r2      @ encoding: [0xd8,0xfa,0x22,0xf4]
@ CHECK: shsub8gt	r4, r8, r2      @ encoding: [0xc8,0xfa,0x22,0xf4]


@------------------------------------------------------------------------------
@ SMLABB/SMLABT/SMLATB/SMLATT
@------------------------------------------------------------------------------
        smlabb r3, r1, r9, r0
        smlabt r5, r6, r4, r1
        smlatb r4, r2, r3, r2
        smlatt r8, r3, r8, r4
        itete gt
        smlabbgt r3, r1, r9, r0
        smlabtle r5, r6, r4, r1
        smlatbgt r4, r2, r3, r2
        smlattle r8, r3, r8, r4

@ CHECK: smlabb	r3, r1, r9, r0          @ encoding: [0x11,0xfb,0x09,0x03]
@ CHECK: smlabt	r5, r6, r4, r1          @ encoding: [0x16,0xfb,0x14,0x15]
@ CHECK: smlatb	r4, r2, r3, r2          @ encoding: [0x12,0xfb,0x23,0x24]
@ CHECK: smlatt	r8, r3, r8, r4          @ encoding: [0x13,0xfb,0x38,0x48]
@ CHECK: itete	gt                      @ encoding: [0xcb,0xbf]
@ CHECK: smlabbgt	r3, r1, r9, r0  @ encoding: [0x11,0xfb,0x09,0x03]
@ CHECK: smlabtle	r5, r6, r4, r1  @ encoding: [0x16,0xfb,0x14,0x15]
@ CHECK: smlatbgt	r4, r2, r3, r2  @ encoding: [0x12,0xfb,0x23,0x24]
@ CHECK: smlattle	r8, r3, r8, r4  @ encoding: [0x13,0xfb,0x38,0x48]


@------------------------------------------------------------------------------
@ SMLAD/SMLADX
@------------------------------------------------------------------------------
        smlad r2, r3, r5, r8
        smladx r2, r3, r5, r8
        itt hi
        smladhi r2, r3, r5, r8
        smladxhi r2, r3, r5, r8

@ CHECK: smlad	r2, r3, r5, r8          @ encoding: [0x23,0xfb,0x05,0x82]
@ CHECK: smladx	r2, r3, r5, r8          @ encoding: [0x23,0xfb,0x15,0x82]
@ CHECK: itt	hi                      @ encoding: [0x84,0xbf]
@ CHECK: smladhi r2, r3, r5, r8         @ encoding: [0x23,0xfb,0x05,0x82]
@ CHECK: smladxhi	r2, r3, r5, r8  @ encoding: [0x23,0xfb,0x15,0x82]


@------------------------------------------------------------------------------
@ SMLAL
@------------------------------------------------------------------------------
        smlal r2, r3, r5, r8
        it eq
        smlaleq r2, r3, r5, r8

@ CHECK: smlal	r2, r3, r5, r8          @ encoding: [0xc5,0xfb,0x08,0x23]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: smlaleq r2, r3, r5, r8         @ encoding: [0xc5,0xfb,0x08,0x23]


@------------------------------------------------------------------------------
@ SMLALBB/SMLALBT/SMLALTB/SMLALTT
@------------------------------------------------------------------------------
        smlalbb r3, r1, r9, r0
        smlalbt r5, r6, r4, r1
        smlaltb r4, r2, r3, r2
        smlaltt r8, r3, r8, r4
        iteet ge
        smlalbbge r3, r1, r9, r0
        smlalbtlt r5, r6, r4, r1
        smlaltblt r4, r2, r3, r2
        smlalttge r8, r3, r8, r4

@ CHECK: smlalbb r3, r1, r9, r0         @ encoding: [0xc9,0xfb,0x80,0x31]
@ CHECK: smlalbt r5, r6, r4, r1         @ encoding: [0xc4,0xfb,0x91,0x56]
@ CHECK: smlaltb r4, r2, r3, r2         @ encoding: [0xc3,0xfb,0xa2,0x42]
@ CHECK: smlaltt r8, r3, r8, r4         @ encoding: [0xc8,0xfb,0xb4,0x83]
@ CHECK: iteet	ge                      @ encoding: [0xad,0xbf]
@ CHECK: smlalbbge	r3, r1, r9, r0  @ encoding: [0xc9,0xfb,0x80,0x31]
@ CHECK: smlalbtlt	r5, r6, r4, r1  @ encoding: [0xc4,0xfb,0x91,0x56]
@ CHECK: smlaltblt	r4, r2, r3, r2  @ encoding: [0xc3,0xfb,0xa2,0x42]
@ CHECK: smlalttge	r8, r3, r8, r4  @ encoding: [0xc8,0xfb,0xb4,0x83]


@------------------------------------------------------------------------------
@ SMLALD/SMLALDX
@------------------------------------------------------------------------------
        smlald r2, r3, r5, r8
        smlaldx r2, r3, r5, r8
        ite eq
        smlaldeq r2, r3, r5, r8
        smlaldxne r2, r3, r5, r8

@ CHECK: smlald	r2, r3, r5, r8          @ encoding: [0xc5,0xfb,0xc8,0x23]
@ CHECK: smlaldx r2, r3, r5, r8         @ encoding: [0xc5,0xfb,0xd8,0x23]
@ CHECK: ite	eq                      @ encoding: [0x0c,0xbf]
@ CHECK: smlaldeq	r2, r3, r5, r8  @ encoding: [0xc5,0xfb,0xc8,0x23]
@ CHECK: smlaldxne	r2, r3, r5, r8  @ encoding: [0xc5,0xfb,0xd8,0x23]


@------------------------------------------------------------------------------
@ SMLAWB/SMLAWT
@------------------------------------------------------------------------------
        smlawb r2, r3, r10, r8
        smlawt r8, r3, r5, r9
        ite eq
        smlawbeq r2, r7, r5, r8
        smlawtne r1, r3, r0, r8

@ CHECK: smlawb	r2, r3, r10, r8         @ encoding: [0x33,0xfb,0x0a,0x82]
@ CHECK: smlawt	r8, r3, r5, r9          @ encoding: [0x33,0xfb,0x15,0x98]
@ CHECK: ite	eq                      @ encoding: [0x0c,0xbf]
@ CHECK: smlawbeq	r2, r7, r5, r8  @ encoding: [0x37,0xfb,0x05,0x82]
@ CHECK: smlawtne	r1, r3, r0, r8  @ encoding: [0x33,0xfb,0x10,0x81]


@------------------------------------------------------------------------------
@ SMLSD/SMLSDX
@------------------------------------------------------------------------------
        smlsd r2, r3, r5, r8
        smlsdx r2, r3, r5, r8
        ite le
        smlsdle r2, r3, r5, r8
        smlsdxgt r2, r3, r5, r8

@ CHECK: smlsd	r2, r3, r5, r8          @ encoding: [0x43,0xfb,0x05,0x82]
@ CHECK: smlsdx	r2, r3, r5, r8          @ encoding: [0x43,0xfb,0x15,0x82]
@ CHECK: ite	le                      @ encoding: [0xd4,0xbf]
@ CHECK: smlsdle	r2, r3, r5, r8  @ encoding: [0x43,0xfb,0x05,0x82]
@ CHECK: smlsdxgt	r2, r3, r5, r8  @ encoding: [0x43,0xfb,0x15,0x82]


@------------------------------------------------------------------------------
@ SMLSLD/SMLSLDX
@------------------------------------------------------------------------------
        smlsld r2, r9, r5, r1
        smlsldx r4, r11, r2, r8
        ite ge
        smlsldge r8, r2, r5, r6
        smlsldxlt r1, r0, r3, r8

@ CHECK: smlsld	r2, r9, r5, r1          @ encoding: [0xd5,0xfb,0xc1,0x29]
@ CHECK: smlsldx	r4, r11, r2, r8 @ encoding: [0xd2,0xfb,0xd8,0x4b]
@ CHECK: ite	ge                      @ encoding: [0xac,0xbf]
@ CHECK: smlsldge	r8, r2, r5, r6  @ encoding: [0xd5,0xfb,0xc6,0x82]
@ CHECK: smlsldxlt	r1, r0, r3, r8  @ encoding: [0xd3,0xfb,0xd8,0x10]


@------------------------------------------------------------------------------
@ SMMLA/SMMLAR
@------------------------------------------------------------------------------
        smmla r1, r2, r3, r4
        smmlar r4, r3, r2, r1
        ite lo
        smmlalo r1, r2, r3, r4
        smmlarcs r4, r3, r2, r1

@ CHECK: smmla	r1, r2, r3, r4          @ encoding: [0x52,0xfb,0x03,0x41]
@ CHECK: smmlar	r4, r3, r2, r1          @ encoding: [0x53,0xfb,0x12,0x14]
@ CHECK: ite	lo                      @ encoding: [0x34,0xbf]
@ CHECK: smmlalo	r1, r2, r3, r4  @ encoding: [0x52,0xfb,0x03,0x41]
@ CHECK: smmlarhs	r4, r3, r2, r1  @ encoding: [0x53,0xfb,0x12,0x14]


@------------------------------------------------------------------------------
@ SMMLS/SMMLSR
@------------------------------------------------------------------------------
        smmls r1, r2, r3, r4
        smmlsr r4, r3, r2, r1
        ite lo
        smmlslo r1, r2, r3, r4
        smmlsrcs r4, r3, r2, r1

@ CHECK: smmls	r1, r2, r3, r4          @ encoding: [0x62,0xfb,0x03,0x41]
@ CHECK: smmlsr	r4, r3, r2, r1          @ encoding: [0x63,0xfb,0x12,0x14]
@ CHECK: ite	lo                      @ encoding: [0x34,0xbf]
@ CHECK: smmlslo	r1, r2, r3, r4  @ encoding: [0x62,0xfb,0x03,0x41]
@ CHECK: smmlsrhs	r4, r3, r2, r1  @ encoding: [0x63,0xfb,0x12,0x14]


@------------------------------------------------------------------------------
@ SMMUL/SMMULR
@------------------------------------------------------------------------------
        smmul r2, r3, r4
        smmulr r3, r2, r1
        ite cc
        smmulcc r2, r3, r4
        smmulrhs r3, r2, r1

@ CHECK: smmul	r2, r3, r4              @ encoding: [0x53,0xfb,0x04,0xf2]
@ CHECK: smmulr	r3, r2, r1              @ encoding: [0x52,0xfb,0x11,0xf3]
@ CHECK: ite	lo                      @ encoding: [0x34,0xbf]
@ CHECK: smmullo	r2, r3, r4      @ encoding: [0x53,0xfb,0x04,0xf2]
@ CHECK: smmulrhs	r3, r2, r1      @ encoding: [0x52,0xfb,0x11,0xf3]


@------------------------------------------------------------------------------
@ SMUAD/SMUADX
@------------------------------------------------------------------------------
        smuad r2, r3, r4
        smuadx r3, r2, r1
        ite lt
        smuadlt r2, r3, r4
        smuadxge r3, r2, r1

@ CHECK: smuad	r2, r3, r4              @ encoding: [0x23,0xfb,0x04,0xf2]
@ CHECK: smuadx	r3, r2, r1              @ encoding: [0x22,0xfb,0x11,0xf3]
@ CHECK: ite	lt                      @ encoding: [0xb4,0xbf]
@ CHECK: smuadlt	r2, r3, r4      @ encoding: [0x23,0xfb,0x04,0xf2]
@ CHECK: smuadxge	r3, r2, r1      @ encoding: [0x22,0xfb,0x11,0xf3]


@------------------------------------------------------------------------------
@ SMULBB/SMULBT/SMULTB/SMULTT
@------------------------------------------------------------------------------
        smulbb r3, r9, r0
        smulbt r5, r4, r1
        smultb r4, r2, r2
        smultt r8, r3, r4
        itete ge
        smulbbge r1, r9, r0
        smulbtlt r5, r6, r4
        smultbge r2, r3, r2
        smulttlt r8, r3, r4

@ CHECK: smulbb	r3, r9, r0              @ encoding: [0x19,0xfb,0x00,0xf3]
@ CHECK: smulbt	r5, r4, r1              @ encoding: [0x14,0xfb,0x11,0xf5]
@ CHECK: smultb	r4, r2, r2              @ encoding: [0x12,0xfb,0x22,0xf4]
@ CHECK: smultt	r8, r3, r4              @ encoding: [0x13,0xfb,0x34,0xf8]
@ CHECK: itete	ge                      @ encoding: [0xab,0xbf]
@ CHECK: smulbbge	r1, r9, r0      @ encoding: [0x19,0xfb,0x00,0xf1]
@ CHECK: smulbtlt	r5, r6, r4      @ encoding: [0x16,0xfb,0x14,0xf5]
@ CHECK: smultbge	r2, r3, r2      @ encoding: [0x13,0xfb,0x22,0xf2]
@ CHECK: smulttlt	r8, r3, r4      @ encoding: [0x13,0xfb,0x34,0xf8]


@------------------------------------------------------------------------------
@ SMULL
@------------------------------------------------------------------------------
        smull r3, r9, r0, r1
        it eq
        smulleq r8, r3, r4, r5

@ CHECK: smull	r3, r9, r0, r1          @ encoding: [0x80,0xfb,0x01,0x39]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: smulleq r8, r3, r4, r5         @ encoding: [0x84,0xfb,0x05,0x83]


@------------------------------------------------------------------------------
@ SMULWB/SMULWT
@------------------------------------------------------------------------------
        smulwb r3, r9, r0
        smulwt r3, r9, r2
        ite gt
        smulwbgt r3, r9, r0
        smulwtle r3, r9, r2

@ CHECK: smulwb	r3, r9, r0              @ encoding: [0x39,0xfb,0x00,0xf3]
@ CHECK: smulwt	r3, r9, r2              @ encoding: [0x39,0xfb,0x12,0xf3]
@ CHECK: ite	gt                      @ encoding: [0xcc,0xbf]
@ CHECK: smulwbgt	r3, r9, r0      @ encoding: [0x39,0xfb,0x00,0xf3]
@ CHECK: smulwtle	r3, r9, r2      @ encoding: [0x39,0xfb,0x12,0xf3]


@------------------------------------------------------------------------------
@ SMUSD/SMUSDX
@------------------------------------------------------------------------------
        smusd r3, r0, r1
        smusdx r3, r9, r2
        ite eq
        smusdeq r8, r3, r2
        smusdxne r7, r4, r3

@ CHECK: smusd	r3, r0, r1              @ encoding: [0x40,0xfb,0x01,0xf3]
@ CHECK: smusdx	r3, r9, r2              @ encoding: [0x49,0xfb,0x12,0xf3]
@ CHECK: ite	eq                      @ encoding: [0x0c,0xbf]
@ CHECK: smusdeq	r8, r3, r2      @ encoding: [0x43,0xfb,0x02,0xf8]
@ CHECK: smusdxne	r7, r4, r3      @ encoding: [0x44,0xfb,0x13,0xf7]


@------------------------------------------------------------------------------
@ SRS
@------------------------------------------------------------------------------
        srsdb sp, #1
        srsia sp, #0

        srsdb sp!, #19
        srsia sp!, #2

        srsea sp, #10
        srsfd sp, #9

        srsea sp!, #5
        srsfd sp!, #5

        srs sp, #5
        srs sp!, #5

@ CHECK: srsdb	sp, #1                  @ encoding: [0x0d,0xe8,0x01,0xc0]
@ CHECK: srsia	sp, #0                  @ encoding: [0x8d,0xe9,0x00,0xc0]
@ CHECK: srsdb	sp!, #19                @ encoding: [0x2d,0xe8,0x13,0xc0]
@ CHECK: srsia	sp!, #2                 @ encoding: [0xad,0xe9,0x02,0xc0]
@ CHECK: srsia	sp, #10                 @ encoding: [0x8d,0xe9,0x0a,0xc0]
@ CHECK: srsdb	sp, #9                  @ encoding: [0x0d,0xe8,0x09,0xc0]
@ CHECK: srsia	sp!, #5                 @ encoding: [0xad,0xe9,0x05,0xc0]
@ CHECK: srsdb	sp!, #5                 @ encoding: [0x2d,0xe8,0x05,0xc0]
@ CHECK: srsia	sp, #5                  @ encoding: [0x8d,0xe9,0x05,0xc0]
@ CHECK: srsia	sp!, #5                 @ encoding: [0xad,0xe9,0x05,0xc0]

        srsdb #1
        srsia #0

        srsdb #19!
        srsia #2!

        srsea #10
        srsfd #9

        srsea #5!
        srsfd #5!

        srs #5
        srs #5!

@ CHECK: srsdb	sp, #1                  @ encoding: [0x0d,0xe8,0x01,0xc0]
@ CHECK: srsia	sp, #0                  @ encoding: [0x8d,0xe9,0x00,0xc0]
@ CHECK: srsdb	sp!, #19                @ encoding: [0x2d,0xe8,0x13,0xc0]
@ CHECK: srsia	sp!, #2                 @ encoding: [0xad,0xe9,0x02,0xc0]
@ CHECK: srsia	sp, #10                 @ encoding: [0x8d,0xe9,0x0a,0xc0]
@ CHECK: srsdb	sp, #9                  @ encoding: [0x0d,0xe8,0x09,0xc0]
@ CHECK: srsia	sp!, #5                 @ encoding: [0xad,0xe9,0x05,0xc0]
@ CHECK: srsdb	sp!, #5                 @ encoding: [0x2d,0xe8,0x05,0xc0]
@ CHECK: srsia	sp, #5                  @ encoding: [0x8d,0xe9,0x05,0xc0]
@ CHECK: srsia	sp!, #5                 @ encoding: [0xad,0xe9,0x05,0xc0]


@------------------------------------------------------------------------------
@ SSAT
@------------------------------------------------------------------------------
        ssat	r8, #1, r10
        ssat	r8, #1, r10, lsl #0
        ssat	r8, #1, r10, lsl #31
        ssat	r8, #1, r10, asr #1

@ CHECK: ssat	r8, #1, r10             @ encoding: [0x0a,0xf3,0x00,0x08]
@ CHECK: ssat	r8, #1, r10             @ encoding: [0x0a,0xf3,0x00,0x08]
@ CHECK: ssat	r8, #1, r10, lsl #31    @ encoding: [0x0a,0xf3,0xc0,0x78]
@ CHECK: ssat	r8, #1, r10, asr #1     @ encoding: [0x2a,0xf3,0x40,0x08]


@------------------------------------------------------------------------------
@ SSAT16
@------------------------------------------------------------------------------
        ssat16	r2, #1, r7
        ssat16	r3, #16, r5

@ CHECK: ssat16	r2, #1, r7              @ encoding: [0x27,0xf3,0x00,0x02]
@ CHECK: ssat16	r3, #16, r5             @ encoding: [0x25,0xf3,0x0f,0x03]


@------------------------------------------------------------------------------
@ SSAX
@------------------------------------------------------------------------------
        ssubaddx r2, r3, r4
        it lt
        ssubaddxlt r2, r3, r4
        ssax r2, r3, r4
        it lt
        ssaxlt r2, r3, r4

@ CHECK: ssax	r2, r3, r4              @ encoding: [0xe3,0xfa,0x04,0xf2]
@ CHECK: it	lt                      @ encoding: [0xb8,0xbf]
@ CHECK: ssaxlt	r2, r3, r4              @ encoding: [0xe3,0xfa,0x04,0xf2]
@ CHECK: ssax	r2, r3, r4              @ encoding: [0xe3,0xfa,0x04,0xf2]
@ CHECK: it	lt                      @ encoding: [0xb8,0xbf]
@ CHECK: ssaxlt	r2, r3, r4              @ encoding: [0xe3,0xfa,0x04,0xf2]


@------------------------------------------------------------------------------
@ SSUB16/SSUB8
@------------------------------------------------------------------------------
        ssub16 r1, r0, r6
        ssub8 r9, r2, r4
        ite ne
        ssub16ne r5, r3, r2
        ssub8eq r5, r1, r2

@ CHECK: ssub16	r1, r0, r6              @ encoding: [0xd0,0xfa,0x06,0xf1]
@ CHECK: ssub8	r9, r2, r4              @ encoding: [0xc2,0xfa,0x04,0xf9]
@ CHECK: ite	ne                      @ encoding: [0x14,0xbf]
@ CHECK: ssub16ne	r5, r3, r2      @ encoding: [0xd3,0xfa,0x02,0xf5]
@ CHECK: ssub8eq	r5, r1, r2      @ encoding: [0xc1,0xfa,0x02,0xf5]


@------------------------------------------------------------------------------
@ STC{L}/STC2{L}
@------------------------------------------------------------------------------
        stc2 p0, c8, [r1, #4]
        stc2 p1, c7, [r2]
        stc2 p2, c6, [r3, #-224]
        stc2 p3, c5, [r4, #-120]!
        stc2 p4, c4, [r5], #16
        stc2 p5, c3, [r6], #-72
        stc2l p6, c2, [r7, #4]
        stc2l p7, c1, [r8]
        stc2l p8, c0, [r9, #-224]
        stc2l p9, c1, [r10, #-120]!
        stc2l p0, c2, [r11], #16
        stc2l p1, c3, [r12], #-72

        stc p12, c4, [r0, #4]
        stc p13, c5, [r1]
        stc p14, c6, [r2, #-224]
        stc p15, c7, [r3, #-120]!
        stc p5, c8, [r4], #16
        stc p4, c9, [r5], #-72
        stcl p3, c10, [r6, #4]
        stcl p2, c11, [r7]
        stcl p1, c12, [r8, #-224]
        stcl p0, c13, [r9, #-120]!
        stcl p6, c14, [r10], #16
        stcl p7, c15, [r11], #-72

        stc2 p2, c8, [r1], { 25 }

@ CHECK: stc2	p0, c8, [r1, #4]        @ encoding: [0x81,0xfd,0x01,0x80]
@ CHECK: stc2	p1, c7, [r2]            @ encoding: [0x82,0xfd,0x00,0x71]
@ CHECK: stc2	p2, c6, [r3, #-224]     @ encoding: [0x03,0xfd,0x38,0x62]
@ CHECK: stc2	p3, c5, [r4, #-120]!    @ encoding: [0x24,0xfd,0x1e,0x53]
@ CHECK: stc2	p4, c4, [r5], #16       @ encoding: [0xa5,0xfc,0x04,0x44]
@ CHECK: stc2	p5, c3, [r6], #-72      @ encoding: [0x26,0xfc,0x12,0x35]
@ CHECK: stc2l	p6, c2, [r7, #4]        @ encoding: [0xc7,0xfd,0x01,0x26]
@ CHECK: stc2l	p7, c1, [r8]            @ encoding: [0xc8,0xfd,0x00,0x17]
@ CHECK: stc2l	p8, c0, [r9, #-224]     @ encoding: [0x49,0xfd,0x38,0x08]
@ CHECK: stc2l	p9, c1, [r10, #-120]!   @ encoding: [0x6a,0xfd,0x1e,0x19]
@ CHECK: stc2l	p0, c2, [r11], #16      @ encoding: [0xeb,0xfc,0x04,0x20]
@ CHECK: stc2l	p1, c3, [r12], #-72     @ encoding: [0x6c,0xfc,0x12,0x31]

@ CHECK: stc	p12, c4, [r0, #4]       @ encoding: [0x80,0xed,0x01,0x4c]
@ CHECK: stc	p13, c5, [r1]           @ encoding: [0x81,0xed,0x00,0x5d]
@ CHECK: stc	p14, c6, [r2, #-224]    @ encoding: [0x02,0xed,0x38,0x6e]
@ CHECK: stc	p15, c7, [r3, #-120]!   @ encoding: [0x23,0xed,0x1e,0x7f]
@ CHECK: stc	p5, c8, [r4], #16       @ encoding: [0xa4,0xec,0x04,0x85]
@ CHECK: stc	p4, c9, [r5], #-72      @ encoding: [0x25,0xec,0x12,0x94]
@ CHECK: stcl	p3, c10, [r6, #4]       @ encoding: [0xc6,0xed,0x01,0xa3]
@ CHECK: stcl	p2, c11, [r7]           @ encoding: [0xc7,0xed,0x00,0xb2]
@ CHECK: stcl	p1, c12, [r8, #-224]    @ encoding: [0x48,0xed,0x38,0xc1]
@ CHECK: stcl	p0, c13, [r9, #-120]!   @ encoding: [0x69,0xed,0x1e,0xd0]
@ CHECK: stcl	p6, c14, [r10], #16     @ encoding: [0xea,0xec,0x04,0xe6]
@ CHECK: stcl	p7, c15, [r11], #-72    @ encoding: [0x6b,0xec,0x12,0xf7]

@ CHECK: stc2	p2, c8, [r1], {25}      @ encoding: [0x81,0xfc,0x19,0x82]


@------------------------------------------------------------------------------
@ STMIA
@------------------------------------------------------------------------------
        stmia.w r4, {r4, r5, r8, r9}
        stmia.w r4, {r5, r6}
        stmia.w r5!, {r3, r8}
        stm.w r4, {r4, r5, r8, r9}
        stm.w r4, {r5, r6}
        stm.w r5!, {r3, r8}
        stm.w r5!, {r1, r2}
        stm.w r2, {r1, r2}

        stmia r4, {r4, r5, r8, r9}
        stmia r4, {r5, r6}
        stmia r5!, {r3, r8}
        stm r4, {r4, r5, r8, r9}
        stm r4, {r5, r6}
        stm r5!, {r3, r8}
        stmea r5!, {r3, r8}

@ CHECK: stm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x84,0xe8,0x30,0x03]
@ CHECK: stm.w	r4, {r5, r6}            @ encoding: [0x84,0xe8,0x60,0x00]
@ CHECK: stm.w	r5!, {r3, r8}           @ encoding: [0xa5,0xe8,0x08,0x01]
@ CHECK: stm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x84,0xe8,0x30,0x03]
@ CHECK: stm.w	r4, {r5, r6}            @ encoding: [0x84,0xe8,0x60,0x00]
@ CHECK: stm.w	r5!, {r3, r8}           @ encoding: [0xa5,0xe8,0x08,0x01]
@ CHECK: stm.w	r5!, {r1, r2}           @ encoding: [0xa5,0xe8,0x06,0x00]
@ CHECK: stm.w	r2, {r1, r2}            @ encoding: [0x82,0xe8,0x06,0x00]

@ CHECK: stm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x84,0xe8,0x30,0x03]
@ CHECK: stm.w	r4, {r5, r6}            @ encoding: [0x84,0xe8,0x60,0x00]
@ CHECK: stm.w	r5!, {r3, r8}           @ encoding: [0xa5,0xe8,0x08,0x01]
@ CHECK: stm.w	r4, {r4, r5, r8, r9}    @ encoding: [0x84,0xe8,0x30,0x03]
@ CHECK: stm.w	r4, {r5, r6}            @ encoding: [0x84,0xe8,0x60,0x00]
@ CHECK: stm.w	r5!, {r3, r8}           @ encoding: [0xa5,0xe8,0x08,0x01]
@ CHECK: stm.w	r5!, {r3, r8}           @ encoding: [0xa5,0xe8,0x08,0x01]


@------------------------------------------------------------------------------
@ STMDB
@------------------------------------------------------------------------------
        stmdb r4, {r4, r5, r8, r9}
        stmdb r4, {r5, r6}
        stmdb r5!, {r3, r8}
        stmea r5!, {r3, r8}
        stmdb.w r5, {r0, r1}

@ CHECK: stmdb	r4, {r4, r5, r8, r9}    @ encoding: [0x04,0xe9,0x30,0x03]
@ CHECK: stmdb	r4, {r5, r6}            @ encoding: [0x04,0xe9,0x60,0x00]
@ CHECK: stmdb	r5!, {r3, r8}           @ encoding: [0x25,0xe9,0x08,0x01]
@ CHECK: stm.w	r5!, {r3, r8}           @ encoding: [0xa5,0xe8,0x08,0x01]
@ CHECK: stmdb	r5, {r0, r1}            @ encoding: [0x05,0xe9,0x03,0x00]


@------------------------------------------------------------------------------
@ STR(immediate)
@------------------------------------------------------------------------------
        str r5, [r5, #-4]
        str r5, [r6, #32]
        str r5, [r6, #33]
        str r5, [r6, #257]
        str.w pc, [r7, #257]
        str r2, [r4, #255]!
        str r8, [sp, #4]!
        str lr, [sp, #-4]!
        str r2, [r4], #255
        str r8, [sp], #4
        str lr, [sp], #-4

@ CHECK: str	r5, [r5, #-4]           @ encoding: [0x45,0xf8,0x04,0x5c]
@ CHECK: str	r5, [r6, #32]           @ encoding: [0x35,0x62]
@ CHECK: str.w	r5, [r6, #33]           @ encoding: [0xc6,0xf8,0x21,0x50]
@ CHECK: str.w	r5, [r6, #257]          @ encoding: [0xc6,0xf8,0x01,0x51]
@ CHECK: str.w	pc, [r7, #257]          @ encoding: [0xc7,0xf8,0x01,0xf1]
@ CHECK: str	r2, [r4, #255]!         @ encoding: [0x44,0xf8,0xff,0x2f]
@ CHECK: str	r8, [sp, #4]!           @ encoding: [0x4d,0xf8,0x04,0x8f]
@ CHECK: str	lr, [sp, #-4]!          @ encoding: [0x4d,0xf8,0x04,0xed]
@ CHECK: str	r2, [r4], #255          @ encoding: [0x44,0xf8,0xff,0x2b]
@ CHECK: str	r8, [sp], #4            @ encoding: [0x4d,0xf8,0x04,0x8b]
@ CHECK: str	lr, [sp], #-4           @ encoding: [0x4d,0xf8,0x04,0xe9]


@------------------------------------------------------------------------------
@ STR(register)
@------------------------------------------------------------------------------
        str r1, [r8, r1]
        str.w r4, [r5, r2]
        str r6, [r0, r2, lsl #3]
        str r8, [r8, r2, lsl #2]
        str r7, [sp, r2, lsl #1]
        str r7, [sp, r2, lsl #0]

@ CHECK: str.w	r1, [r8, r1]            @ encoding: [0x48,0xf8,0x01,0x10]
@ CHECK: str.w	r4, [r5, r2]            @ encoding: [0x45,0xf8,0x02,0x40]
@ CHECK: str.w	r6, [r0, r2, lsl #3]    @ encoding: [0x40,0xf8,0x32,0x60]
@ CHECK: str.w	r8, [r8, r2, lsl #2]    @ encoding: [0x48,0xf8,0x22,0x80]
@ CHECK: str.w	r7, [sp, r2, lsl #1]    @ encoding: [0x4d,0xf8,0x12,0x70]
@ CHECK: str.w	r7, [sp, r2]            @ encoding: [0x4d,0xf8,0x02,0x70]


@------------------------------------------------------------------------------
@ STRB(immediate)
@------------------------------------------------------------------------------
        strb r5, [r5, #-4]
        strb r5, [r6, #32]
        strb r5, [r6, #33]
        strb r5, [r6, #257]
        strb.w lr, [r7, #257]
        strb r5, [r8, #255]!
        strb r2, [r5, #4]!
        strb r1, [r4, #-4]!
        strb lr, [r3], #255
        strb r9, [r2], #4
        strb r3, [sp], #-4
        strb r4, [r8, #-0]!
        strb r1, [r0], #-0

@ CHECK: strb	r5, [r5, #-4]           @ encoding: [0x05,0xf8,0x04,0x5c]
@ CHECK: strb.w	r5, [r6, #32]           @ encoding: [0x86,0xf8,0x20,0x50]
@ CHECK: strb.w	r5, [r6, #33]           @ encoding: [0x86,0xf8,0x21,0x50]
@ CHECK: strb.w	r5, [r6, #257]          @ encoding: [0x86,0xf8,0x01,0x51]
@ CHECK: strb.w	lr, [r7, #257]          @ encoding: [0x87,0xf8,0x01,0xe1]
@ CHECK: strb	r5, [r8, #255]!         @ encoding: [0x08,0xf8,0xff,0x5f]
@ CHECK: strb	r2, [r5, #4]!           @ encoding: [0x05,0xf8,0x04,0x2f]
@ CHECK: strb	r1, [r4, #-4]!          @ encoding: [0x04,0xf8,0x04,0x1d]
@ CHECK: strb	lr, [r3], #255          @ encoding: [0x03,0xf8,0xff,0xeb]
@ CHECK: strb	r9, [r2], #4            @ encoding: [0x02,0xf8,0x04,0x9b]
@ CHECK: strb	r3, [sp], #-4           @ encoding: [0x0d,0xf8,0x04,0x39]
@ CHECK: strb	r4, [r8, #-0]!          @ encoding: [0x08,0xf8,0x00,0x4d]
@ CHECK: strb	r1, [r0], #-0           @ encoding: [0x00,0xf8,0x00,0x19]


@------------------------------------------------------------------------------
@ STRB(register)
@------------------------------------------------------------------------------
        strb r1, [r8, r1]
        strb.w r4, [r5, r2]
        strb r6, [r0, r2, lsl #3]
        strb r8, [r8, r2, lsl #2]
        strb r7, [sp, r2, lsl #1]
        strb r7, [sp, r2, lsl #0]

@ CHECK: strb.w	r1, [r8, r1]            @ encoding: [0x08,0xf8,0x01,0x10]
@ CHECK: strb.w	r4, [r5, r2]            @ encoding: [0x05,0xf8,0x02,0x40]
@ CHECK: strb.w	r6, [r0, r2, lsl #3]    @ encoding: [0x00,0xf8,0x32,0x60]
@ CHECK: strb.w	r8, [r8, r2, lsl #2]    @ encoding: [0x08,0xf8,0x22,0x80]
@ CHECK: strb.w	r7, [sp, r2, lsl #1]    @ encoding: [0x0d,0xf8,0x12,0x70]
@ CHECK: strb.w	r7, [sp, r2]            @ encoding: [0x0d,0xf8,0x02,0x70]


@------------------------------------------------------------------------------
@ STRBT
@------------------------------------------------------------------------------
        strbt r1, [r2]
        strbt r1, [r8, #0]
        strbt r1, [r8, #3]
        strbt r1, [r8, #255]

@ CHECK: strbt	r1, [r2]                @ encoding: [0x02,0xf8,0x00,0x1e]
@ CHECK: strbt	r1, [r8]                @ encoding: [0x08,0xf8,0x00,0x1e]
@ CHECK: strbt	r1, [r8, #3]            @ encoding: [0x08,0xf8,0x03,0x1e]
@ CHECK: strbt	r1, [r8, #255]          @ encoding: [0x08,0xf8,0xff,0x1e]


@------------------------------------------------------------------------------
@ STRD
@------------------------------------------------------------------------------
        strd r3, r5, [r6, #24]
        strd r3, r5, [r6, #24]!
        strd r3, r5, [r6], #4
        strd r3, r5, [r6], #-8
        strd r3, r5, [r6]
        strd r8, r1, [r3, #0]
        strd r0, r1, [r2, #-0]
        strd r0, r1, [r2, #-0]!
        strd r0, r1, [r2], #-0
        strd r0, r1, [r2, #256]
        strd r0, r1, [r2, #256]!
        strd r0, r1, [r2], #256

@ CHECK: strd	r3, r5, [r6, #24]       @ encoding: [0xc6,0xe9,0x06,0x35]
@ CHECK: strd	r3, r5, [r6, #24]!      @ encoding: [0xe6,0xe9,0x06,0x35]
@ CHECK: strd	r3, r5, [r6], #4        @ encoding: [0xe6,0xe8,0x01,0x35]
@ CHECK: strd	r3, r5, [r6], #-8       @ encoding: [0x66,0xe8,0x02,0x35]
@ CHECK: strd	r3, r5, [r6]            @ encoding: [0xc6,0xe9,0x00,0x35]
@ CHECK: strd	r8, r1, [r3]            @ encoding: [0xc3,0xe9,0x00,0x81]
@ CHECK: strd   r0, r1, [r2, #-0]       @ encoding: [0x42,0xe9,0x00,0x01]
@ CHECK: strd   r0, r1, [r2, #-0]!      @ encoding: [0x62,0xe9,0x00,0x01]
@ CHECK: strd   r0, r1, [r2], #-0       @ encoding: [0x62,0xe8,0x00,0x01]
@ CHECK: strd	r0, r1, [r2, #256]      @ encoding: [0xc2,0xe9,0x40,0x01]
@ CHECK: strd	r0, r1, [r2, #256]!     @ encoding: [0xe2,0xe9,0x40,0x01]
@ CHECK: strd	r0, r1, [r2], #256      @ encoding: [0xe2,0xe8,0x40,0x01]


@------------------------------------------------------------------------------
@ STREX/STREXB/STREXH/STREXD
@------------------------------------------------------------------------------
        strex r1, r8, [r4]
        strex r8, r2, [r4, #0]
        strex r2, r12, [sp, #128]
        strexb r5, r1, [r7]
        strexh r9, r7, [r12]
        strexd r9, r3, r6, [r4]

@ CHECK: strex	r1, r8, [r4]            @ encoding: [0x44,0xe8,0x00,0x81]
@ CHECK: strex	r8, r2, [r4]            @ encoding: [0x44,0xe8,0x00,0x28]
@ CHECK: strex	r2, r12, [sp, #128]     @ encoding: [0x4d,0xe8,0x20,0xc2]
@ CHECK: strexb	r5, r1, [r7]            @ encoding: [0xc7,0xe8,0x45,0x1f]
@ CHECK: strexh	r9, r7, [r12]           @ encoding: [0xcc,0xe8,0x59,0x7f]
@ CHECK: strexd	r9, r3, r6, [r4]        @ encoding: [0xc4,0xe8,0x79,0x36]


@------------------------------------------------------------------------------
@ STRH(immediate)
@------------------------------------------------------------------------------
        strh r5, [r5, #-4]
        strh r5, [r6, #32]
        strh r5, [r6, #33]
        strh r5, [r6, #257]
        strh.w lr, [r7, #257]
        strh r5, [r8, #255]!
        strh r2, [r5, #4]!
        strh r1, [r4, #-4]!
        strh lr, [r3], #255
        strh r9, [r2], #4
        strh r3, [sp], #-4

@ CHECK: strh	r5, [r5, #-4]           @ encoding: [0x25,0xf8,0x04,0x5c]
@ CHECK: strh	r5, [r6, #32]           @ encoding: [0x35,0x84]
@ CHECK: strh.w	r5, [r6, #33]           @ encoding: [0xa6,0xf8,0x21,0x50]
@ CHECK: strh.w	r5, [r6, #257]          @ encoding: [0xa6,0xf8,0x01,0x51]
@ CHECK: strh.w	lr, [r7, #257]          @ encoding: [0xa7,0xf8,0x01,0xe1]
@ CHECK: strh	r5, [r8, #255]!         @ encoding: [0x28,0xf8,0xff,0x5f]
@ CHECK: strh	r2, [r5, #4]!           @ encoding: [0x25,0xf8,0x04,0x2f]
@ CHECK: strh	r1, [r4, #-4]!          @ encoding: [0x24,0xf8,0x04,0x1d]
@ CHECK: strh	lr, [r3], #255          @ encoding: [0x23,0xf8,0xff,0xeb]
@ CHECK: strh	r9, [r2], #4            @ encoding: [0x22,0xf8,0x04,0x9b]
@ CHECK: strh	r3, [sp], #-4           @ encoding: [0x2d,0xf8,0x04,0x39]


@------------------------------------------------------------------------------
@ STRH(register)
@------------------------------------------------------------------------------
        strh r1, [r8, r1]
        strh.w r4, [r5, r2]
        strh r6, [r0, r2, lsl #3]
        strh r8, [r8, r2, lsl #2]
        strh r7, [sp, r2, lsl #1]
        strh r7, [sp, r2, lsl #0]

@ CHECK: strh.w	r1, [r8, r1]            @ encoding: [0x28,0xf8,0x01,0x10]
@ CHECK: strh.w	r4, [r5, r2]            @ encoding: [0x25,0xf8,0x02,0x40]
@ CHECK: strh.w	r6, [r0, r2, lsl #3]    @ encoding: [0x20,0xf8,0x32,0x60]
@ CHECK: strh.w	r8, [r8, r2, lsl #2]    @ encoding: [0x28,0xf8,0x22,0x80]
@ CHECK: strh.w	r7, [sp, r2, lsl #1]    @ encoding: [0x2d,0xf8,0x12,0x70]
@ CHECK: strh.w	r7, [sp, r2]            @ encoding: [0x2d,0xf8,0x02,0x70]


@------------------------------------------------------------------------------
@ STRHT
@------------------------------------------------------------------------------
        strht r1, [r2]
        strht r1, [r8, #0]
        strht r1, [r8, #3]
        strht r1, [r8, #255]

@ CHECK: strht	r1, [r2]                @ encoding: [0x22,0xf8,0x00,0x1e]
@ CHECK: strht	r1, [r8]                @ encoding: [0x28,0xf8,0x00,0x1e]
@ CHECK: strht	r1, [r8, #3]            @ encoding: [0x28,0xf8,0x03,0x1e]
@ CHECK: strht	r1, [r8, #255]          @ encoding: [0x28,0xf8,0xff,0x1e]


@------------------------------------------------------------------------------
@ STRT
@------------------------------------------------------------------------------
        strt r1, [r2]
        strt r1, [r8, #0]
        strt r1, [r8, #3]
        strt r1, [r8, #255]

@ CHECK: strt	r1, [r2]                @ encoding: [0x42,0xf8,0x00,0x1e]
@ CHECK: strt	r1, [r8]                @ encoding: [0x48,0xf8,0x00,0x1e]
@ CHECK: strt	r1, [r8, #3]            @ encoding: [0x48,0xf8,0x03,0x1e]
@ CHECK: strt	r1, [r8, #255]          @ encoding: [0x48,0xf8,0xff,0x1e]


@------------------------------------------------------------------------------
@ SUB (immediate)
@------------------------------------------------------------------------------
        itet eq
        subeq r1, r2, #4
        subwne r5, r3, #1023
        subeq r4, r5, #293
        sub r2, sp, #1024
        sub r2, r8, #0xff00
        sub r2, r3, #257
        subw r2, r3, #257
        sub r12, r6, #0x100
        subw r12, r6, #0x100
        subs r1, r2, #0x1f0
	sub r2, #1
        sub r0, r0, #32
        subs r2, r2, #56
        subs r2, #56

@ CHECK: itet	eq                      @ encoding: [0x0a,0xbf]
@ CHECK: subeq	r1, r2, #4              @ encoding: [0x11,0x1f]
@ CHECK: subwne	r5, r3, #1023           @ encoding: [0xa3,0xf2,0xff,0x35]
@ CHECK: subweq	r4, r5, #293            @ encoding: [0xa5,0xf2,0x25,0x14]
@ CHECK: sub.w	r2, sp, #1024           @ encoding: [0xad,0xf5,0x80,0x62]
@ CHECK: sub.w	r2, r8, #65280          @ encoding: [0xa8,0xf5,0x7f,0x42]
@ CHECK: subw	r2, r3, #257            @ encoding: [0xa3,0xf2,0x01,0x12]
@ CHECK: subw	r2, r3, #257            @ encoding: [0xa3,0xf2,0x01,0x12]
@ CHECK: sub.w	r12, r6, #256           @ encoding: [0xa6,0xf5,0x80,0x7c]
@ CHECK: subw	r12, r6, #256           @ encoding: [0xa6,0xf2,0x00,0x1c]
@ CHECK: subs.w	r1, r2, #496            @ encoding: [0xb2,0xf5,0xf8,0x71]
@ CHECK: sub.w	r2, r2, #1              @ encoding: [0xa2,0xf1,0x01,0x02]
@ CHECK: sub.w	r0, r0, #32             @ encoding: [0xa0,0xf1,0x20,0x00]
@ CHECK: subs	r2, #56                 @ encoding: [0x38,0x3a]
@ CHECK: subs	r2, #56                 @ encoding: [0x38,0x3a]


@------------------------------------------------------------------------------
@ SUB (register)
@------------------------------------------------------------------------------
        sub r4, r5, r6
        sub r4, r5, r6, lsl #5
        sub r4, r5, r6, lsr #5
        sub.w r4, r5, r6, lsr #5
        sub r4, r5, r6, asr #5
        sub r4, r5, r6, ror #5
        sub.w r5, r2, r12, rrx
        sub r2, sp, ip
        sub sp, sp, ip
        sub sp, ip
        sub.w r2, sp, ip
        sub.w sp, sp, ip
        sub.w sp, ip

@ CHECK: sub.w	r4, r5, r6              @ encoding: [0xa5,0xeb,0x06,0x04]
@ CHECK: sub.w	r4, r5, r6, lsl #5      @ encoding: [0xa5,0xeb,0x46,0x14]
@ CHECK: sub.w	r4, r5, r6, lsr #5      @ encoding: [0xa5,0xeb,0x56,0x14]
@ CHECK: sub.w	r4, r5, r6, lsr #5      @ encoding: [0xa5,0xeb,0x56,0x14]
@ CHECK: sub.w	r4, r5, r6, asr #5      @ encoding: [0xa5,0xeb,0x66,0x14]
@ CHECK: sub.w	r4, r5, r6, ror #5      @ encoding: [0xa5,0xeb,0x76,0x14]
@ CHECK: sub.w r5, r2, r12, rrx         @ encoding: [0xa2,0xeb,0x3c,0x05]
@ CHECK: sub.w	r2, sp, r12             @ encoding: [0xad,0xeb,0x0c,0x02]
@ CHECK: sub.w	sp, sp, r12             @ encoding: [0xad,0xeb,0x0c,0x0d]
@ CHECK: sub.w	sp, sp, r12             @ encoding: [0xad,0xeb,0x0c,0x0d]
@ CHECK: sub.w	r2, sp, r12             @ encoding: [0xad,0xeb,0x0c,0x02]
@ CHECK: sub.w	sp, sp, r12             @ encoding: [0xad,0xeb,0x0c,0x0d]
@ CHECK: sub.w	sp, sp, r12             @ encoding: [0xad,0xeb,0x0c,0x0d]


@------------------------------------------------------------------------------
@ SVC
@------------------------------------------------------------------------------
        svc #0
        it eq
        svceq #255
        it ne
        swine #33
        itt eq
        svceq #0
        svceq #1

@ CHECK: svc	#0                      @ encoding: [0x00,0xdf]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: svceq	#255                    @ encoding: [0xff,0xdf]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: svcne	#33                     @ encoding: [0x21,0xdf]
@ CHECK: itt    eq                      @ encoding: [0x04,0xbf]
@ CHECK: svceq  #0                      @ encoding: [0x00,0xdf]
@ CHECK: svceq  #1                      @ encoding: [0x01,0xdf]


@------------------------------------------------------------------------------
@ SXTAB
@------------------------------------------------------------------------------
        sxtab r2, r3, r4
        sxtab r4, r5, r6, ror #0
        it lt
        sxtablt r6, r2, r9, ror #8
        sxtab r5, r1, r4, ror #16
        sxtab r7, r8, r3, ror #24

@ CHECK: sxtab	r2, r3, r4              @ encoding: [0x43,0xfa,0x84,0xf2]
@ CHECK: sxtab	r4, r5, r6              @ encoding: [0x45,0xfa,0x86,0xf4]
@ CHECK: it	lt                      @ encoding: [0xb8,0xbf]
@ CHECK: sxtablt r6, r2, r9, ror #8     @ encoding: [0x42,0xfa,0x99,0xf6]
@ CHECK: sxtab	r5, r1, r4, ror #16     @ encoding: [0x41,0xfa,0xa4,0xf5]
@ CHECK: sxtab	r7, r8, r3, ror #24     @ encoding: [0x48,0xfa,0xb3,0xf7]


@------------------------------------------------------------------------------
@ SXTAB16
@------------------------------------------------------------------------------
        sxtab16 r6, r2, r7, ror #0
        sxtab16 r3, r5, r8, ror #8
        sxtab16 r3, r2, r1, ror #16
        ite ne
        sxtab16ne r0, r1, r4
        sxtab16eq r1, r2, r3, ror #24

@ CHECK: sxtab16 r6, r2, r7             @ encoding: [0x22,0xfa,0x87,0xf6]
@ CHECK: sxtab16 r3, r5, r8, ror #8     @ encoding: [0x25,0xfa,0x98,0xf3]
@ CHECK: sxtab16 r3, r2, r1, ror #16    @ encoding: [0x22,0xfa,0xa1,0xf3]
@ CHECK: ite	ne                      @ encoding: [0x14,0xbf]
@ CHECK: sxtab16ne r0, r1, r4           @ encoding: [0x21,0xfa,0x84,0xf0]
@ CHECK: sxtab16eq r1, r2, r3, ror #24  @ encoding: [0x22,0xfa,0xb3,0xf1]


@------------------------------------------------------------------------------
@ SXTAH
@------------------------------------------------------------------------------
        sxtah r1, r3, r9
        sxtah r3, r8, r3, ror #8
        sxtah r9, r3, r3, ror #24
        ite hi
        sxtahhi r6, r1, r6, ror #0
        sxtahls r2, r2, r4, ror #16

@ CHECK: sxtah	r1, r3, r9              @ encoding: [0x03,0xfa,0x89,0xf1]
@ CHECK: sxtah	r3, r8, r3, ror #8      @ encoding: [0x08,0xfa,0x93,0xf3]
@ CHECK: sxtah	r9, r3, r3, ror #24     @ encoding: [0x03,0xfa,0xb3,0xf9]
@ CHECK: ite	hi                      @ encoding: [0x8c,0xbf]
@ CHECK: sxtahhi r6, r1, r6             @ encoding: [0x01,0xfa,0x86,0xf6]
@ CHECK: sxtahls r2, r2, r4, ror #16    @ encoding: [0x02,0xfa,0xa4,0xf2]


@------------------------------------------------------------------------------
@ SXTB
@------------------------------------------------------------------------------
        sxtb r5, r6, ror #0
        sxtb r6, r9, ror #8
        sxtb r8, r3, ror #24
        ite ge
        sxtbge r2, r4
        sxtblt r5, r1, ror #16
        sxtb.w  r7, r8

@ CHECK: sxtb	r5, r6                  @ encoding: [0x75,0xb2]
@ CHECK: sxtb.w	r6, r9, ror #8          @ encoding: [0x4f,0xfa,0x99,0xf6]
@ CHECK: sxtb.w	r8, r3, ror #24         @ encoding: [0x4f,0xfa,0xb3,0xf8]
@ CHECK: ite	ge                      @ encoding: [0xac,0xbf]
@ CHECK: sxtbge	r2, r4                  @ encoding: [0x62,0xb2]
@ CHECK: sxtblt.w	r5, r1, ror #16 @ encoding: [0x4f,0xfa,0xa1,0xf5]
@ CHECK: sxtb.w	r7, r8                  @ encoding: [0x4f,0xfa,0x88,0xf7]


@------------------------------------------------------------------------------
@ SXTB16
@------------------------------------------------------------------------------
        sxtb16 r1, r4
        sxtb16 r6, r7, ror #0
        sxtb16 r3, r1, ror #16
        ite cs
        sxtb16cs r3, r5, ror #8
        sxtb16lo r2, r3, ror #24

@ CHECK: sxtb16	r1, r4                  @ encoding: [0x2f,0xfa,0x84,0xf1]
@ CHECK: sxtb16	r6, r7                  @ encoding: [0x2f,0xfa,0x87,0xf6]
@ CHECK: sxtb16	r3, r1, ror #16         @ encoding: [0x2f,0xfa,0xa1,0xf3]
@ CHECK: ite	hs                      @ encoding: [0x2c,0xbf]
@ CHECK: sxtb16hs	r3, r5, ror #8  @ encoding: [0x2f,0xfa,0x95,0xf3]
@ CHECK: sxtb16lo	r2, r3, ror #24 @ encoding: [0x2f,0xfa,0xb3,0xf2]


@------------------------------------------------------------------------------
@ SXTH
@------------------------------------------------------------------------------
        sxth r1, r6, ror #0
        sxth r3, r8, ror #8
        sxth r9, r3, ror #24
        itt ne
        sxthne r3, r9
        sxthne r2, r2, ror #16
        sxth.w  r7, r8

@ CHECK: sxth	r1, r6                  @ encoding: [0x31,0xb2]
@ CHECK: sxth.w	r3, r8, ror #8          @ encoding: [0x0f,0xfa,0x98,0xf3]
@ CHECK: sxth.w	r9, r3, ror #24         @ encoding: [0x0f,0xfa,0xb3,0xf9]
@ CHECK: itt	ne                      @ encoding: [0x1c,0xbf]
@ CHECK: sxthne.w	r3, r9          @ encoding: [0x0f,0xfa,0x89,0xf3]
@ CHECK: sxthne.w	r2, r2, ror #16 @ encoding: [0x0f,0xfa,0xa2,0xf2]
@ CHECK: sxth.w	r7, r8                  @ encoding: [0x0f,0xfa,0x88,0xf7]


@------------------------------------------------------------------------------
@ SXTB
@------------------------------------------------------------------------------
        sxtb r5, r6, ror #0
        sxtb.w r6, r9, ror #8
        sxtb r8, r3, ror #24
        ite ge
        sxtbge r2, r4
        sxtblt r5, r1, ror #16

@ CHECK: sxtb	r5, r6                  @ encoding: [0x75,0xb2]
@ CHECK: sxtb.w	r6, r9, ror #8          @ encoding: [0x4f,0xfa,0x99,0xf6]
@ CHECK: sxtb.w	r8, r3, ror #24         @ encoding: [0x4f,0xfa,0xb3,0xf8]
@ CHECK: ite	ge                      @ encoding: [0xac,0xbf]
@ CHECK: sxtbge	r2, r4                  @ encoding: [0x62,0xb2]
@ CHECK: sxtblt.w	r5, r1, ror #16 @ encoding: [0x4f,0xfa,0xa1,0xf5]


@------------------------------------------------------------------------------
@ SXTB16
@------------------------------------------------------------------------------
        sxtb16 r1, r4
        sxtb16 r6, r7, ror #0
        sxtb16 r3, r1, ror #16
        ite cs
        sxtb16cs r3, r5, ror #8
        sxtb16lo r2, r3, ror #24

@ CHECK: sxtb16	r1, r4                  @ encoding: [0x2f,0xfa,0x84,0xf1]
@ CHECK: sxtb16	r6, r7                  @ encoding: [0x2f,0xfa,0x87,0xf6]
@ CHECK: sxtb16	r3, r1, ror #16         @ encoding: [0x2f,0xfa,0xa1,0xf3]
@ CHECK: ite	hs                      @ encoding: [0x2c,0xbf]
@ CHECK: sxtb16hs	r3, r5, ror #8  @ encoding: [0x2f,0xfa,0x95,0xf3]
@ CHECK: sxtb16lo	r2, r3, ror #24 @ encoding: [0x2f,0xfa,0xb3,0xf2]


@------------------------------------------------------------------------------
@ SXTH
@------------------------------------------------------------------------------
        sxth r1, r6, ror #0
        sxth.w r3, r8, ror #8
        sxth r9, r3, ror #24
        itt ne
        sxthne r3, r9
        sxthne r2, r2, ror #16

@ CHECK: sxth	r1, r6                  @ encoding: [0x31,0xb2]
@ CHECK: sxth.w	r3, r8, ror #8          @ encoding: [0x0f,0xfa,0x98,0xf3]
@ CHECK: sxth.w	r9, r3, ror #24         @ encoding: [0x0f,0xfa,0xb3,0xf9]
@ CHECK: itt	ne                      @ encoding: [0x1c,0xbf]
@ CHECK: sxthne.w	r3, r9          @ encoding: [0x0f,0xfa,0x89,0xf3]
@ CHECK: sxthne.w	r2, r2, ror #16 @ encoding: [0x0f,0xfa,0xa2,0xf2]


@------------------------------------------------------------------------------
@ TBB/TBH
@------------------------------------------------------------------------------
        tbb [r3, r8]
        tbh [r3, r8, lsl #1]
        it eq
        tbbeq [r3, r8]
        it cs
        tbhcs [r3, r8, lsl #1]

@ CHECK: tbb	[r3, r8]                @ encoding: [0xd3,0xe8,0x08,0xf0]
@ CHECK: tbh	[r3, r8, lsl #1]        @ encoding: [0xd3,0xe8,0x18,0xf0]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: tbbeq	[r3, r8]                @ encoding: [0xd3,0xe8,0x08,0xf0]
@ CHECK: it	hs                      @ encoding: [0x28,0xbf]
@ CHECK: tbhhs	[r3, r8, lsl #1]        @ encoding: [0xd3,0xe8,0x18,0xf0]


@------------------------------------------------------------------------------
@ TEQ
@------------------------------------------------------------------------------
        teq r5, #0xf000
        teq r4, r5
        teq r4, r5, lsl #5
        teq r4, r5, lsr #5
        teq r4, r5, lsr #5
        teq r4, r5, asr #5
        teq r4, r5, ror #5

@ CHECK: teq.w	r5, #61440              @ encoding: [0x95,0xf4,0x70,0x4f]
@ CHECK: teq.w	r4, r5                  @ encoding: [0x94,0xea,0x05,0x0f]
@ CHECK: teq.w	r4, r5, lsl #5          @ encoding: [0x94,0xea,0x45,0x1f]
@ CHECK: teq.w	r4, r5, lsr #5          @ encoding: [0x94,0xea,0x55,0x1f]
@ CHECK: teq.w	r4, r5, lsr #5          @ encoding: [0x94,0xea,0x55,0x1f]
@ CHECK: teq.w	r4, r5, asr #5          @ encoding: [0x94,0xea,0x65,0x1f]
@ CHECK: teq.w	r4, r5, ror #5          @ encoding: [0x94,0xea,0x75,0x1f]


@------------------------------------------------------------------------------
@ TST
@------------------------------------------------------------------------------
        tst r5, #0xf000
        tst r2, r5
        tst r3, r12, lsl #5
        tst r4, r11, lsr #4
        tst r5, r10, lsr #12
        tst r6, r9, asr #30
        tst r7, r8, ror #2

@ CHECK: tst.w	r5, #61440              @ encoding: [0x15,0xf4,0x70,0x4f]
@ CHECK: tst	r2, r5                  @ encoding: [0x2a,0x42]
@ CHECK: tst.w	r3, r12, lsl #5         @ encoding: [0x13,0xea,0x4c,0x1f]
@ CHECK: tst.w	r4, r11, lsr #4         @ encoding: [0x14,0xea,0x1b,0x1f]
@ CHECK: tst.w	r5, r10, lsr #12        @ encoding: [0x15,0xea,0x1a,0x3f]
@ CHECK: tst.w	r6, r9, asr #30         @ encoding: [0x16,0xea,0xa9,0x7f]
@ CHECK: tst.w	r7, r8, ror #2          @ encoding: [0x17,0xea,0xb8,0x0f]


@------------------------------------------------------------------------------
@ UADD16/UADD8
@------------------------------------------------------------------------------
        uadd16 r1, r2, r3
        uadd8 r1, r2, r3
        ite gt
        uadd16gt r1, r2, r3
        uadd8le r1, r2, r3

@ CHECK: uadd16	r1, r2, r3              @ encoding: [0x92,0xfa,0x43,0xf1]
@ CHECK: uadd8	r1, r2, r3              @ encoding: [0x82,0xfa,0x43,0xf1]
@ CHECK: ite	gt                      @ encoding: [0xcc,0xbf]
@ CHECK: uadd16gt	r1, r2, r3      @ encoding: [0x92,0xfa,0x43,0xf1]
@ CHECK: uadd8le	r1, r2, r3      @ encoding: [0x82,0xfa,0x43,0xf1]


@------------------------------------------------------------------------------
@ UASX
@------------------------------------------------------------------------------
        uasx r9, r12, r0
        it eq
        uasxeq r9, r12, r0
        uaddsubx r9, r12, r0
        it eq
        uaddsubxeq r9, r12, r0

@ CHECK: uasx	r9, r12, r0             @ encoding: [0xac,0xfa,0x40,0xf9]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: uasxeq	r9, r12, r0             @ encoding: [0xac,0xfa,0x40,0xf9]
@ CHECK: uasx	r9, r12, r0             @ encoding: [0xac,0xfa,0x40,0xf9]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: uasxeq	r9, r12, r0             @ encoding: [0xac,0xfa,0x40,0xf9]


@------------------------------------------------------------------------------
@ UBFX
@------------------------------------------------------------------------------
        ubfx r4, r5, #16, #1
        it gt
        ubfxgt r4, r5, #16, #16

@ CHECK: ubfx	r4, r5, #16, #1         @ encoding: [0xc5,0xf3,0x00,0x44]
@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: ubfxgt	r4, r5, #16, #16        @ encoding: [0xc5,0xf3,0x0f,0x44]


@------------------------------------------------------------------------------
@ UHADD16/UHADD8
@------------------------------------------------------------------------------
        uhadd16 r4, r8, r2
        uhadd8 r4, r8, r2
        itt gt
        uhadd16gt r4, r8, r2
        uhadd8gt r4, r8, r2

@ CHECK: uhadd16	r4, r8, r2      @ encoding: [0x98,0xfa,0x62,0xf4]
@ CHECK: uhadd8	r4, r8, r2              @ encoding: [0x88,0xfa,0x62,0xf4]
@ CHECK: itt	gt                      @ encoding: [0xc4,0xbf]
@ CHECK: uhadd16gt	r4, r8, r2      @ encoding: [0x98,0xfa,0x62,0xf4]
@ CHECK: uhadd8gt	r4, r8, r2      @ encoding: [0x88,0xfa,0x62,0xf4]


@------------------------------------------------------------------------------
@ UHASX/UHSAX
@------------------------------------------------------------------------------
        uhasx r4, r1, r5
        uhsax r5, r6, r6
        itt gt
        uhasxgt r6, r9, r8
        uhsaxgt r7, r8, r12
        uhaddsubx r4, r1, r5
        uhsubaddx r5, r6, r6
        itt gt
        uhaddsubxgt r6, r9, r8
        uhsubaddxgt r7, r8, r12

@ CHECK: uhasx	r4, r1, r5              @ encoding: [0xa1,0xfa,0x65,0xf4]
@ CHECK: uhsax	r5, r6, r6              @ encoding: [0xe6,0xfa,0x66,0xf5]
@ CHECK: itt	gt                      @ encoding: [0xc4,0xbf]
@ CHECK: uhasxgt r6, r9, r8             @ encoding: [0xa9,0xfa,0x68,0xf6]
@ CHECK: uhsaxgt r7, r8, r12            @ encoding: [0xe8,0xfa,0x6c,0xf7]
@ CHECK: uhasx	r4, r1, r5              @ encoding: [0xa1,0xfa,0x65,0xf4]
@ CHECK: uhsax	r5, r6, r6              @ encoding: [0xe6,0xfa,0x66,0xf5]
@ CHECK: itt	gt                      @ encoding: [0xc4,0xbf]
@ CHECK: uhasxgt r6, r9, r8             @ encoding: [0xa9,0xfa,0x68,0xf6]
@ CHECK: uhsaxgt r7, r8, r12            @ encoding: [0xe8,0xfa,0x6c,0xf7]


@------------------------------------------------------------------------------
@ UHSUB16/UHSUB8
@------------------------------------------------------------------------------
        uhsub16 r5, r8, r3
        uhsub8 r1, r7, r6
        itt lt
        uhsub16lt r4, r9, r12
        uhsub8lt r3, r1, r5

@ CHECK: uhsub16	r5, r8, r3      @ encoding: [0xd8,0xfa,0x63,0xf5]
@ CHECK: uhsub8	r1, r7, r6              @ encoding: [0xc7,0xfa,0x66,0xf1]
@ CHECK: itt	lt                      @ encoding: [0xbc,0xbf]
@ CHECK: uhsub16lt	r4, r9, r12     @ encoding: [0xd9,0xfa,0x6c,0xf4]
@ CHECK: uhsub8lt	r3, r1, r5      @ encoding: [0xc1,0xfa,0x65,0xf3]


@------------------------------------------------------------------------------
@ UMAAL
@------------------------------------------------------------------------------
        umaal r3, r4, r5, r6
        it lt
        umaallt r3, r4, r5, r6

@ CHECK: umaal	r3, r4, r5, r6          @ encoding: [0xe5,0xfb,0x66,0x34]
@ CHECK: it	lt                      @ encoding: [0xb8,0xbf]
@ CHECK: umaallt	r3, r4, r5, r6  @ encoding: [0xe5,0xfb,0x66,0x34]


@------------------------------------------------------------------------------
@ UMLAL
@------------------------------------------------------------------------------
        umlal r2, r4, r6, r8
        it gt
        umlalgt r6, r1, r2, r6

@ CHECK: umlal	r2, r4, r6, r8          @ encoding: [0xe6,0xfb,0x08,0x24]
@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: umlalgt	r6, r1, r2, r6  @ encoding: [0xe2,0xfb,0x06,0x61]


@------------------------------------------------------------------------------
@ UMULL
@------------------------------------------------------------------------------
        umull r2, r4, r6, r8
        it gt
        umullgt r6, r1, r2, r6

@ CHECK: umull	r2, r4, r6, r8          @ encoding: [0xa6,0xfb,0x08,0x24]
@ CHECK: it	gt                      @ encoding: [0xc8,0xbf]
@ CHECK: umullgt	r6, r1, r2, r6  @ encoding: [0xa2,0xfb,0x06,0x61]


@------------------------------------------------------------------------------
@ UQADD16/UQADD8
@------------------------------------------------------------------------------
        uqadd16 r1, r2, r3
        uqadd8 r3, r4, r8
        ite gt
        uqadd16gt r4, r7, r9
        uqadd8le r8, r1, r2

@ CHECK: uqadd16	r1, r2, r3      @ encoding: [0x92,0xfa,0x53,0xf1]
@ CHECK: uqadd8	r3, r4, r8              @ encoding: [0x84,0xfa,0x58,0xf3]
@ CHECK: ite	gt                      @ encoding: [0xcc,0xbf]
@ CHECK: uqadd16gt	r4, r7, r9      @ encoding: [0x97,0xfa,0x59,0xf4]
@ CHECK: uqadd8le	r8, r1, r2      @ encoding: [0x81,0xfa,0x52,0xf8]


@------------------------------------------------------------------------------
@ UQASX/UQSAX
@------------------------------------------------------------------------------
        uqasx r1, r2, r3
        uqsax r3, r4, r8
        ite gt
        uqasxgt r4, r7, r9
        uqsaxle r8, r1, r2

        uqaddsubx r1, r2, r3
        uqsubaddx r3, r4, r8
        ite gt
        uqaddsubxgt r4, r7, r9
        uqsubaddxle r8, r1, r2

@ CHECK: uqasx	r1, r2, r3              @ encoding: [0xa2,0xfa,0x53,0xf1]
@ CHECK: uqsax	r3, r4, r8              @ encoding: [0xe4,0xfa,0x58,0xf3]
@ CHECK: ite	gt                      @ encoding: [0xcc,0xbf]
@ CHECK: uqasxgt r4, r7, r9             @ encoding: [0xa7,0xfa,0x59,0xf4]
@ CHECK: uqsaxle r8, r1, r2             @ encoding: [0xe1,0xfa,0x52,0xf8]

@ CHECK: uqasx	r1, r2, r3              @ encoding: [0xa2,0xfa,0x53,0xf1]
@ CHECK: uqsax	r3, r4, r8              @ encoding: [0xe4,0xfa,0x58,0xf3]
@ CHECK: ite	gt                      @ encoding: [0xcc,0xbf]
@ CHECK: uqasxgt r4, r7, r9             @ encoding: [0xa7,0xfa,0x59,0xf4]
@ CHECK: uqsaxle r8, r1, r2             @ encoding: [0xe1,0xfa,0x52,0xf8]


@------------------------------------------------------------------------------
@ UQSUB16/UQSUB8
@------------------------------------------------------------------------------
        uqsub8 r8, r2, r9
        uqsub16 r1, r9, r7
        ite gt
        uqsub8gt r3, r1, r6
        uqsub16le r4, r6, r4

@ CHECK: uqsub8	r8, r2, r9              @ encoding: [0xc2,0xfa,0x59,0xf8]
@ CHECK: uqsub16 r1, r9, r7             @ encoding: [0xd9,0xfa,0x57,0xf1]
@ CHECK: ite	gt                      @ encoding: [0xcc,0xbf]
@ CHECK: uqsub8gt	r3, r1, r6      @ encoding: [0xc1,0xfa,0x56,0xf3]
@ CHECK: uqsub16le	r4, r6, r4      @ encoding: [0xd6,0xfa,0x54,0xf4]


@------------------------------------------------------------------------------
@ UQSUB16/UQSUB8
@------------------------------------------------------------------------------
        usad8 r1, r9, r7
        usada8 r8, r2, r9, r12
        ite gt
        usada8gt r3, r1, r6, r9
        usad8le r4, r6, r4

@ CHECK: usad8	r1, r9, r7              @ encoding: [0x79,0xfb,0x07,0xf1]
@ CHECK: usada8	r8, r2, r9, r12         @ encoding: [0x72,0xfb,0x09,0xc8]
@ CHECK: ite	gt                      @ encoding: [0xcc,0xbf]
@ CHECK: usada8gt	r3, r1, r6, r9  @ encoding: [0x71,0xfb,0x06,0x93]
@ CHECK: usad8le	r4, r6, r4      @ encoding: [0x76,0xfb,0x04,0xf4]


@------------------------------------------------------------------------------
@ USAT
@------------------------------------------------------------------------------
        usat	r8, #1, r10
        usat	r8, #4, r10, lsl #0
        usat	r8, #5, r10, lsl #31
        usat	r8, #16, r10, asr #1

@ CHECK: usat	r8, #1, r10             @ encoding: [0x8a,0xf3,0x01,0x08]
@ CHECK: usat	r8, #4, r10             @ encoding: [0x8a,0xf3,0x04,0x08]
@ CHECK: usat	r8, #5, r10, lsl #31    @ encoding: [0x8a,0xf3,0xc5,0x78]
@ CHECK: usat	r8, #16, r10, asr #1    @ encoding: [0xaa,0xf3,0x50,0x08]


@------------------------------------------------------------------------------
@ USAT16
@------------------------------------------------------------------------------
        usat16	r2, #2, r7
        usat16	r3, #15, r5

@ CHECK: usat16	r2, #2, r7              @ encoding: [0xa7,0xf3,0x02,0x02]
@ CHECK: usat16	r3, #15, r5             @ encoding: [0xa5,0xf3,0x0f,0x03]


@------------------------------------------------------------------------------
@ USAX
@------------------------------------------------------------------------------
        usax r2, r3, r4
        it ne
        usaxne r6, r1, r9
        usubaddx r2, r3, r4
        it ne
        usubaddxne r6, r1, r9

@ CHECK: usax	r2, r3, r4              @ encoding: [0xe3,0xfa,0x44,0xf2]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: usaxne	r6, r1, r9              @ encoding: [0xe1,0xfa,0x49,0xf6]
@ CHECK: usax	r2, r3, r4              @ encoding: [0xe3,0xfa,0x44,0xf2]
@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: usaxne	r6, r1, r9              @ encoding: [0xe1,0xfa,0x49,0xf6]


@------------------------------------------------------------------------------
@ USUB16/USUB8
@------------------------------------------------------------------------------
        usub16 r4, r2, r7
        usub8 r1, r8, r5
        ite hi
        usub16hi r1, r1, r3
        usub8ls r9, r2, r3

@ CHECK: usub16	r4, r2, r7              @ encoding: [0xd2,0xfa,0x47,0xf4]
@ CHECK: usub8	r1, r8, r5              @ encoding: [0xc8,0xfa,0x45,0xf1]
@ CHECK: ite	hi                      @ encoding: [0x8c,0xbf]
@ CHECK: usub16hi	r1, r1, r3      @ encoding: [0xd1,0xfa,0x43,0xf1]
@ CHECK: usub8ls	r9, r2, r3      @ encoding: [0xc2,0xfa,0x43,0xf9]


@------------------------------------------------------------------------------
@ UXTAB
@------------------------------------------------------------------------------
        uxtab r2, r3, r4
        uxtab r4, r5, r6, ror #0
        it lt
        uxtablt r6, r2, r9, ror #8
        uxtab r5, r1, r4, ror #16
        uxtab r7, r8, r3, ror #24

@ CHECK: uxtab	r2, r3, r4              @ encoding: [0x53,0xfa,0x84,0xf2]
@ CHECK: uxtab	r4, r5, r6              @ encoding: [0x55,0xfa,0x86,0xf4]
@ CHECK: it	lt                      @ encoding: [0xb8,0xbf]
@ CHECK: uxtablt r6, r2, r9, ror #8     @ encoding: [0x52,0xfa,0x99,0xf6]
@ CHECK: uxtab	r5, r1, r4, ror #16     @ encoding: [0x51,0xfa,0xa4,0xf5]
@ CHECK: uxtab	r7, r8, r3, ror #24     @ encoding: [0x58,0xfa,0xb3,0xf7]


@------------------------------------------------------------------------------
@ UXTAB16
@------------------------------------------------------------------------------
        it ge
        uxtab16ge r0, r1, r4
        uxtab16 r6, r2, r7, ror #0
        uxtab16 r3, r5, r8, ror #8
        uxtab16 r3, r2, r1, ror #16
        it eq
        uxtab16eq r1, r2, r3, ror #24

@ CHECK: it	ge                      @ encoding: [0xa8,0xbf]
@ CHECK: uxtab16ge	r0, r1, r4      @ encoding: [0x31,0xfa,0x84,0xf0]
@ CHECK: uxtab16 r6, r2, r7             @ encoding: [0x32,0xfa,0x87,0xf6]
@ CHECK: uxtab16 r3, r5, r8, ror #8     @ encoding: [0x35,0xfa,0x98,0xf3]
@ CHECK: uxtab16 r3, r2, r1, ror #16    @ encoding: [0x32,0xfa,0xa1,0xf3]
@ CHECK: it	eq                      @ encoding: [0x08,0xbf]
@ CHECK: uxtab16eq r1, r2, r3, ror #24  @ encoding: [0x32,0xfa,0xb3,0xf1]


@------------------------------------------------------------------------------
@ UXTAH
@------------------------------------------------------------------------------
        uxtah r1, r3, r9
        it hi
        uxtahhi r6, r1, r6, ror #0
        uxtah r3, r8, r3, ror #8
        it lo
        uxtahlo r2, r2, r4, ror #16
        uxtah r9, r3, r3, ror #24

@ CHECK: uxtah	r1, r3, r9              @ encoding: [0x13,0xfa,0x89,0xf1]
@ CHECK: it	hi                      @ encoding: [0x88,0xbf]
@ CHECK: uxtahhi r6, r1, r6             @ encoding: [0x11,0xfa,0x86,0xf6]
@ CHECK: uxtah	r3, r8, r3, ror #8      @ encoding: [0x18,0xfa,0x93,0xf3]
@ CHECK: it	lo                      @ encoding: [0x38,0xbf]
@ CHECK: uxtahlo r2, r2, r4, ror #16    @ encoding: [0x12,0xfa,0xa4,0xf2]
@ CHECK: uxtah	r9, r3, r3, ror #24     @ encoding: [0x13,0xfa,0xb3,0xf9]


@------------------------------------------------------------------------------
@ UXTB
@------------------------------------------------------------------------------
        it ge
        uxtbge r2, r4
        uxtb r5, r6, ror #0
        uxtb r6, r9, ror #8
        it cc
        uxtbcc r5, r1, ror #16
        uxtb r8, r3, ror #24
        uxtb.w  r7, r8

@ CHECK: it	ge                      @ encoding: [0xa8,0xbf]
@ CHECK: uxtbge	r2, r4                  @ encoding: [0xe2,0xb2]
@ CHECK: uxtb	r5, r6                  @ encoding: [0xf5,0xb2]
@ CHECK: uxtb.w	r6, r9, ror #8          @ encoding: [0x5f,0xfa,0x99,0xf6]
@ CHECK: it	lo                      @ encoding: [0x38,0xbf]
@ CHECK: uxtblo.w	r5, r1, ror #16 @ encoding: [0x5f,0xfa,0xa1,0xf5]
@ CHECK: uxtb.w	r8, r3, ror #24         @ encoding: [0x5f,0xfa,0xb3,0xf8]
@ CHECK: uxtb.w	r7, r8                  @ encoding: [0x5f,0xfa,0x88,0xf7]


@------------------------------------------------------------------------------
@ UXTB16
@------------------------------------------------------------------------------
        uxtb16 r1, r4
        uxtb16 r6, r7, ror #0
        it cs
        uxtb16cs r3, r5, ror #8
        uxtb16 r3, r1, ror #16
        it ge
        uxtb16ge r2, r3, ror #24

@ CHECK: uxtb16	r1, r4                  @ encoding: [0x3f,0xfa,0x84,0xf1]
@ CHECK: uxtb16	r6, r7                  @ encoding: [0x3f,0xfa,0x87,0xf6]
@ CHECK: it	hs                      @ encoding: [0x28,0xbf]
@ CHECK: uxtb16hs	r3, r5, ror #8  @ encoding: [0x3f,0xfa,0x95,0xf3]
@ CHECK: uxtb16	r3, r1, ror #16         @ encoding: [0x3f,0xfa,0xa1,0xf3]
@ CHECK: it	ge                      @ encoding: [0xa8,0xbf]
@ CHECK: uxtb16ge	r2, r3, ror #24 @ encoding: [0x3f,0xfa,0xb3,0xf2]


@------------------------------------------------------------------------------
@ UXTH
@------------------------------------------------------------------------------
        it ne
        uxthne r3, r9
        uxth r1, r6, ror #0
        uxth r3, r8, ror #8
        it le
        uxthle r2, r2, ror #16
        uxth r9, r3, ror #24
        uxth.w  r7, r8

@ CHECK: it	ne                      @ encoding: [0x18,0xbf]
@ CHECK: uxthne.w	r3, r9          @ encoding: [0x1f,0xfa,0x89,0xf3]
@ CHECK: uxth	r1, r6                  @ encoding: [0xb1,0xb2]
@ CHECK: uxth.w	r3, r8, ror #8          @ encoding: [0x1f,0xfa,0x98,0xf3]
@ CHECK: it	le                      @ encoding: [0xd8,0xbf]
@ CHECK: uxthle.w	r2, r2, ror #16 @ encoding: [0x1f,0xfa,0xa2,0xf2]
@ CHECK: uxth.w	r9, r3, ror #24         @ encoding: [0x1f,0xfa,0xb3,0xf9]
@ CHECK: uxth.w	r7, r8                  @ encoding: [0x1f,0xfa,0x88,0xf7]

@------------------------------------------------------------------------------
@ WFE/WFI/YIELD/HINT
@------------------------------------------------------------------------------
        wfe
        wfi
        yield
        itet lt
        wfelt
        wfige
        yieldlt
        hint.w #4
        hint.w #3
        hint.w #2
        hint.w #1
        hint.w #0
        hint #4
        hint #3
        hint #2
        hint #1
        hint #0

        itet lt
        hintlt #15
        hintge #16
        hintlt #239

@ CHECK: wfe                            @ encoding: [0x20,0xbf]
@ CHECK: wfi                            @ encoding: [0x30,0xbf]
@ CHECK: yield                          @ encoding: [0x10,0xbf]
@ CHECK: itet	lt                      @ encoding: [0xb6,0xbf]
@ CHECK: wfelt                          @ encoding: [0x20,0xbf]
@ CHECK: wfige                          @ encoding: [0x30,0xbf]
@ CHECK: yieldlt                        @ encoding: [0x10,0xbf]
@ CHECK: sev.w                          @ encoding: [0xaf,0xf3,0x04,0x80]
@ CHECK: wfi.w                          @ encoding: [0xaf,0xf3,0x03,0x80]
@ CHECK: wfe.w                          @ encoding: [0xaf,0xf3,0x02,0x80]
@ CHECK: yield.w                        @ encoding: [0xaf,0xf3,0x01,0x80]
@ CHECK: nop.w                          @ encoding: [0xaf,0xf3,0x00,0x80]
@ CHECK: sev                            @ encoding: [0x40,0xbf]
@ CHECK: wfi                            @ encoding: [0x30,0xbf]
@ CHECK: wfe                            @ encoding: [0x20,0xbf]
@ CHECK: yield                          @ encoding: [0x10,0xbf]
@ CHECK: nop                            @ encoding: [0x00,0xbf]

@ CHECK: itet	lt                      @ encoding: [0xb6,0xbf]
@ CHECK: hintlt #15                     @ encoding: [0xf0,0xbf]
@ CHECK: hintge.w #16                   @ encoding: [0xaf,0xf3,0x10,0x80]
@ CHECK: hintlt.w #239                  @ encoding: [0xaf,0xf3,0xef,0x80]

@------------------------------------------------------------------------------
@ Unallocated wide/narrow hints
@------------------------------------------------------------------------------
        hint #7
        hint.w #7
@ CHECK: hint #7                        @ encoding: [0x70,0xbf]
@ CHECK: hint.w #7                      @ encoding: [0xaf,0xf3,0x07,0x80]

@------------------------------------------------------------------------------
@ Alternate syntax for LDR*(literal) encodings
@------------------------------------------------------------------------------
        ldrb r11, [pc, #22]
        ldrh r11, [pc, #22]
        ldrsb r11, [pc, #22]
        ldrsh r11, [pc, #22]
        ldr.w r11, [pc, #22]
        ldrb.w r11, [pc, #22]
        ldrh.w r11, [pc, #22]
        ldrsb.w r11, [pc, #22]
        ldrsh.w r11, [pc, #22]

@ CHECK: ldrb.w r11, [pc, #22]        @ encoding: [0x9f,0xf8,0x16,0xb0]
@ CHECK: ldrh.w r11, [pc, #22]        @ encoding: [0xbf,0xf8,0x16,0xb0]
@ CHECK: ldrsb.w r11, [pc, #22]       @ encoding: [0x9f,0xf9,0x16,0xb0]
@ CHECK: ldrsh.w r11, [pc, #22]       @ encoding: [0xbf,0xf9,0x16,0xb0]
@ CHECK: ldr.w r11, [pc, #22]         @ encoding: [0xdf,0xf8,0x16,0xb0]
@ CHECK: ldrb.w r11, [pc, #22]        @ encoding: [0x9f,0xf8,0x16,0xb0]
@ CHECK: ldrh.w r11, [pc, #22]        @ encoding: [0xbf,0xf8,0x16,0xb0]
@ CHECK: ldrsb.w r11, [pc, #22]       @ encoding: [0x9f,0xf9,0x16,0xb0]
@ CHECK: ldrsh.w r11, [pc, #22]       @ encoding: [0xbf,0xf9,0x16,0xb0]

        ldr r11, [pc, #-22]
        ldrb r11, [pc, #-22]
        ldrh r11, [pc, #-22]
        ldrsb r11, [pc, #-22]
        ldrsh r11, [pc, #-22]
        ldr.w r11, [pc, #-22]
        ldrb.w r11, [pc, #-22]
        ldrh.w r11, [pc, #-22]
        ldrsb.w r11, [pc, #-22]
        ldrsh.w r11, [pc, #-22]

@ CHECK: ldr.w	r11, [pc, #-22]         @ encoding: [0x5f,0xf8,0x16,0xb0]
@ CHECK: ldrb.w	r11, [pc, #-22]         @ encoding: [0x1f,0xf8,0x16,0xb0]
@ CHECK: ldrh.w	r11, [pc, #-22]         @ encoding: [0x3f,0xf8,0x16,0xb0]
@ CHECK: ldrsb.w r11, [pc, #-22]        @ encoding: [0x1f,0xf9,0x16,0xb0]
@ CHECK: ldrsh.w r11, [pc, #-22]        @ encoding: [0x3f,0xf9,0x16,0xb0]
@ CHECK: ldr.w	r11, [pc, #-22]         @ encoding: [0x5f,0xf8,0x16,0xb0]
@ CHECK: ldrb.w	r11, [pc, #-22]         @ encoding: [0x1f,0xf8,0x16,0xb0]
@ CHECK: ldrh.w	r11, [pc, #-22]         @ encoding: [0x3f,0xf8,0x16,0xb0]
@ CHECK: ldrsb.w r11, [pc, #-22]        @ encoding: [0x1f,0xf9,0x16,0xb0]
@ CHECK: ldrsh.w r11, [pc, #-22]        @ encoding: [0x3f,0xf9,0x16,0xb0]

@ rdar://12596361
         ldr r1, [pc, #12]
@ CHECK: ldr r1, [pc, #12]              @ encoding: [0x03,0x49]

@ rdar://14214063
         subs pc, lr, #4
@ CHECK: subs pc, lr, #4                @ encoding: [0xde,0xf3,0x04,0x8f]
