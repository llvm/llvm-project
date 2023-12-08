@ RUN: llvm-mc -triple=armv7-apple-darwin -mcpu=cortex-a8 -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=armebv7-unknown-unknown -mcpu=cortex-a8 -show-encoding < %s | FileCheck --check-prefix=CHECK-BE %s
  .syntax unified
  .globl _func

@ Check that the assembler can handle the documented syntax from the ARM ARM.
@ For complex constructs like shifter operands, check more thoroughly for them
@ once then spot check that following instructions accept the form generally.
@ This gives us good coverage while keeping the overall size of the test
@ more reasonable.

_func:
@ CHECK: _func

@------------------------------------------------------------------------------
@ ADC (immediate)
@------------------------------------------------------------------------------
        adc r1, r2, #0xf
        adc r1, r2, $0xf
        adc r1, r2, 0xf
        adc r7, r8, #(0xff << 16)
        adc r7, r8, #-2147483638
        adc r7, r8, #42, #2
        adc r7, r8, #40, #2
        adc r7, r8, $40, $2
        adc r7, r8, 40, 2
        adc r7, r8, (2 * 20), (1 << 1)
        adc r1, r2, #0xf0
        adc r1, r2, #0xf00
        adc r1, r2, #0xf000
        adc r1, r2, #0xf0000
        adc r1, r2, #0xf00000
        adc r1, r2, #0xf000000
        adc r1, r2, #0xf0000000
        adc r1, r2, #0xf000000f
        adcs r1, r2, #0xf00
        adcs r7, r8, #40, #2
        adcseq r1, r2, #0xf00
        adceq r1, r2, #0xf00

@ CHECK: adc	r1, r2, #15             @ encoding: [0x0f,0x10,0xa2,0xe2]
@ CHECK: adc	r1, r2, #15             @ encoding: [0x0f,0x10,0xa2,0xe2]
@ CHECK: adc	r1, r2, #15             @ encoding: [0x0f,0x10,0xa2,0xe2]
@ CHECK: adc	r7, r8, #16711680       @ encoding: [0xff,0x78,0xa8,0xe2]
@ CHECK: adc    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0xa8,0xe2]
@ CHECK: adc    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0xa8,0xe2]
@ CHECK: adc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xa8,0xe2]
@ CHECK: adc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xa8,0xe2]
@ CHECK: adc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xa8,0xe2]
@ CHECK: adc	r7, r8, #40, #2         @ encoding: [0x28,0x71,0xa8,0xe2]
@ CHECK: adc	r1, r2, #240            @ encoding: [0xf0,0x10,0xa2,0xe2]
@ CHECK: adc	r1, r2, #3840           @ encoding: [0x0f,0x1c,0xa2,0xe2]
@ CHECK: adc	r1, r2, #61440          @ encoding: [0x0f,0x1a,0xa2,0xe2]
@ CHECK: adc	r1, r2, #983040         @ encoding: [0x0f,0x18,0xa2,0xe2]
@ CHECK: adc	r1, r2, #15728640       @ encoding: [0x0f,0x16,0xa2,0xe2]
@ CHECK: adc	r1, r2, #251658240      @ encoding: [0x0f,0x14,0xa2,0xe2]
@ CHECK: adc	r1, r2, #-268435456     @ encoding: [0x0f,0x12,0xa2,0xe2]
@ CHECK: adc	r1, r2, #-268435441     @ encoding: [0xff,0x12,0xa2,0xe2]
@ CHECK: adcs	r1, r2, #3840           @ encoding: [0x0f,0x1c,0xb2,0xe2]
@ CHECK: adcs   r7, r8, #40, #2         @ encoding: [0x28,0x71,0xb8,0xe2]
@ CHECK: adcseq	r1, r2, #3840           @ encoding: [0x0f,0x1c,0xb2,0x02]
@ CHECK: adceq	r1, r2, #3840           @ encoding: [0x0f,0x1c,0xa2,0x02]

@------------------------------------------------------------------------------
@ ADC (register)
@ ADC (shifted register)
@------------------------------------------------------------------------------
        adc r4, r5, r6
        @ Constant shifts
        adc r4, r5, r6, lsl #1
        adc r4, r5, r6, lsl #31
        adc r4, r5, r6, lsr #1
        adc r4, r5, r6, lsr #31
        adc r4, r5, r6, lsr #32
        adc r4, r5, r6, asr #1
        adc r4, r5, r6, asr #31
        adc r4, r5, r6, asr #32
        adc r4, r5, r6, ror #1
        adc r4, r5, r6, ror #31

        @ Register shifts
        adc r6, r7, r8, lsl r9
        adc r6, r7, r8, lsr r9
        adc r6, r7, r8, asr r9
        adc r6, r7, r8, ror r9
        adc r4, r5, r6, rrx

        @ Destination register is optional
        adc r5, r6
        adc r4, r5, lsl #1
        adc r4, r5, lsl #31
        adc r4, r5, lsr #1
        adc r4, r5, lsr #31
        adc r4, r5, lsr #32
        adc r4, r5, asr #1
        adc r4, r5, asr #31
        adc r4, r5, asr #32
        adc r4, r5, ror #1
        adc r4, r5, ror #31
        adc r4, r5, rrx
        adc r6, r7, lsl r9
        adc r6, r7, lsr r9
        adc r6, r7, asr r9
        adc r6, r7, ror r9
        adc r4, r5, rrx

@ CHECK: adc	r4, r5, r6              @ encoding: [0x06,0x40,0xa5,0xe0]

@ CHECK: adc	r4, r5, r6, lsl #1      @ encoding: [0x86,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, lsl #31     @ encoding: [0x86,0x4f,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, lsr #1      @ encoding: [0xa6,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, lsr #31     @ encoding: [0xa6,0x4f,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, lsr #32     @ encoding: [0x26,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, asr #1      @ encoding: [0xc6,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, asr #31     @ encoding: [0xc6,0x4f,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, asr #32     @ encoding: [0x46,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, ror #1      @ encoding: [0xe6,0x40,0xa5,0xe0]
@ CHECK: adc	r4, r5, r6, ror #31     @ encoding: [0xe6,0x4f,0xa5,0xe0]

@ CHECK: adc	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0xa7,0xe0]
@ CHECK: adc	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0xa7,0xe0]
@ CHECK: adc	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0xa7,0xe0]
@ CHECK: adc	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0xa7,0xe0]
@ CHECK: adc	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0xa5,0xe0]

@ CHECK: adc	r5, r5, r6              @ encoding: [0x06,0x50,0xa5,0xe0]
@ CHECK: adc	r4, r4, r5, lsl #1      @ encoding: [0x85,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, lsl #31     @ encoding: [0x85,0x4f,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, lsr #1      @ encoding: [0xa5,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, lsr #31     @ encoding: [0xa5,0x4f,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, lsr #32     @ encoding: [0x25,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, asr #1      @ encoding: [0xc5,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, asr #31     @ encoding: [0xc5,0x4f,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, asr #32     @ encoding: [0x45,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, ror #1      @ encoding: [0xe5,0x40,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, ror #31     @ encoding: [0xe5,0x4f,0xa4,0xe0]
@ CHECK: adc	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0xa4,0xe0]
@ CHECK: adc	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0xa6,0xe0]
@ CHECK: adc	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0xa6,0xe0]
@ CHECK: adc	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0xa6,0xe0]
@ CHECK: adc	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0xa6,0xe0]
@ CHECK: adc	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0xa4,0xe0]


@------------------------------------------------------------------------------
@ ADR
@------------------------------------------------------------------------------
Lback:
        adr r2, Lback
        adr r3, Lforward
Lforward:
        adr	r2, #3
        adr	r2, #-3

@ CHECK: Lback:
@ CHECK: adr	r2, Lback    @ encoding: [A,0x20'A',0x0f'A',0xe2'A']
@ CHECK:  @   fixup A - offset: 0, value: Lback, kind: fixup_arm_adr_pcrel_12
@ CHECK-BE: adr	r2, Lback    @ encoding: [0xe2'A',0x0f'A',0x20'A',A]
@ CHECK-BE:  @   fixup A - offset: 0, value: Lback, kind: fixup_arm_adr_pcrel_12
@ CHECK: adr	r3, Lforward @ encoding: [A,0x30'A',0x0f'A',0xe2'A']
@ CHECK:  @   fixup A - offset: 0, value: Lforward, kind: fixup_arm_adr_pcrel_12
@ CHECK-BE: adr	r3, Lforward @ encoding: [0xe2'A',0x0f'A',0x30'A',A]
@ CHECK-BE:  @   fixup A - offset: 0, value: Lforward, kind: fixup_arm_adr_pcrel_12
@ CHECK: Lforward:
@ CHECK: adr	r2, #3                  @ encoding: [0x03,0x20,0x8f,0xe2]
@ CHECK: adr	r2, #-3                 @ encoding: [0x03,0x20,0x4f,0xe2]

        adr r1, #-0x0
        adr r1, #-0x12000000
        adr r1, #-0x80000001
        adr r1, #0x12000000
        adr r1, #0x80000001

@ CHECK: adr	r1, #-0                 @ encoding: [0x00,0x10,0x4f,0xe2]
@ CHECK: adr	r1, #-301989888         @ encoding: [0x12,0x14,0x4f,0xe2]
@ CHECK: adr	r1, #2147483647         @ encoding: [0x06,0x11,0x4f,0xe2]
@ CHECK: adr	r1, #301989888          @ encoding: [0x12,0x14,0x8f,0xe2]
@ CHECK: adr	r1, #-2147483647        @ encoding: [0x06,0x11,0x8f,0xe2]

@------------------------------------------------------------------------------
@ ADD
@------------------------------------------------------------------------------
        add r4, r5, #0xf000
        add r4, r5, $0xf000
        add r4, r5, 0xf000
        add r4, r5, -0xf000
	add r7, r8, #(0xff << 16)
        add r7, r8, #-2147483638
        add r7, r8, #42, #2
        add r7, r8, #40, #2
        add r7, r8, $40, $2
        add r7, r8, 40, 2
        add r7, r8, (2 * 20), (1 << 1)
        add r4, r5, r6
        add r4, r5, r6, lsl #5
        add r4, r5, r6, lsr #5
        add r4, r5, r6, lsr #5
        add r4, r5, r6, asr #5
        add r4, r5, r6, ror #5
        add r6, r7, r8, lsl r9
        add r4, r4, r3, asl r9
        add r6, r7, r8, lsr r9
        add r6, r7, r8, asr r9
        add r6, r7, r8, ror r9
        add r4, r5, r6, rrx

        @ destination register is optional
        add r5, #0xf000
        add r5, $0xf000
        add r5, 0xf000
        add r5, -0xf000
	add r7, #(0xff << 16)
        add r7, #-2147483638
        add r7, #42, #2
        add r7, #40, #2
        add r7, $40, $2
        add r7, 40, 2
        add r7, (2 * 20), (1 << 1)
        add r4, r5
        add r4, r5, lsl #5
        add r4, r5, lsr #5
        add r4, r5, lsr #5
        add r4, r5, asr #5
        add r4, r5, ror #5
        add r6, r7, lsl r9
        add r6, r7, lsr r9
        add r6, r7, asr r9
        add r6, r7, ror r9
        add r4, r5, rrx

        add r0, #-4
        add r4, r5, #-21
        add r0, pc, #0xc0000000
        addseq r0,pc,#0xc0000000


        add r0, pc, #(Lback - .)

@ CHECK: add	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x85,0xe2]
@ CHECK: add	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x85,0xe2]
@ CHECK: add	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x85,0xe2]
@ CHECK: sub	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x45,0xe2]
@ CHECK: add	r7, r8, #16711680       @ encoding: [0xff,0x78,0x88,0xe2]
@ CHECK: add    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x88,0xe2]
@ CHECK: add    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x88,0xe2]
@ CHECK: add    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x88,0xe2]
@ CHECK: add    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x88,0xe2]
@ CHECK: add    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x88,0xe2]
@ CHECK: add    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x88,0xe2]
@ CHECK: add	r4, r5, r6              @ encoding: [0x06,0x40,0x85,0xe0]
@ CHECK: add	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x85,0xe0]
@ CHECK: add	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x85,0xe0]
@ CHECK: add	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x85,0xe0]
@ CHECK: add	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x85,0xe0]
@ CHECK: add	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x85,0xe0]
@ CHECK: add	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x87,0xe0]
@ CHECK: add	r4, r4, r3, lsl r9      @ encoding: [0x13,0x49,0x84,0xe0]
@ CHECK: add	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x87,0xe0]
@ CHECK: add	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x87,0xe0]
@ CHECK: add	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x87,0xe0]
@ CHECK: add	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x85,0xe0]

@ CHECK: add	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0xe2]
@ CHECK: add	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0xe2]
@ CHECK: add	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0xe2]
@ CHECK: sub	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x45,0xe2]
@ CHECK: add	r7, r7, #16711680       @ encoding: [0xff,0x78,0x87,0xe2]
@ CHECK: add	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x87,0xe2]
@ CHECK: add	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x87,0xe2]
@ CHECK: add	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x87,0xe2]
@ CHECK: add	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x87,0xe2]
@ CHECK: add	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x87,0xe2]
@ CHECK: add	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x87,0xe2]
@ CHECK: add	r4, r4, r5              @ encoding: [0x05,0x40,0x84,0xe0]
@ CHECK: add	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x84,0xe0]
@ CHECK: add	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x84,0xe0]
@ CHECK: add	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x84,0xe0]
@ CHECK: add	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x84,0xe0]
@ CHECK: add	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x84,0xe0]
@ CHECK: add	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x86,0xe0]
@ CHECK: add	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x86,0xe0]
@ CHECK: add	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x86,0xe0]
@ CHECK: add	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x86,0xe0]
@ CHECK: add	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x84,0xe0]

@ CHECK: sub	r0, r0, #4              @ encoding: [0x04,0x00,0x40,0xe2]
@ CHECK: sub	r4, r5, #21             @ encoding: [0x15,0x40,0x45,0xe2]
@ CHECK: adr	r0, #-1073741824        @ encoding: [0x03,0x01,0x8f,0xe2]
@ CHECK: addseq r0, pc, #-1073741824    @ encoding: [0x03,0x01,0x9f,0x02]
@ CHECK:        Ltmp0:
@ CHECK-NEXT:   Ltmp1:
@ CHECK-NEXT:   adr	r0, (Ltmp1+8)+(Lback-Ltmp0) @ encoding: [A,A,0x0f'A',0xe2'A']
@ CHECK-NEXT:                           @   fixup A - offset: 0, value: (Ltmp1+8)+(Lback-Ltmp0), kind: fixup_arm_adr_pcrel_12

    @ Test right shift by 32, which is encoded as 0
    add r3, r1, r2, lsr #32
    add r3, r1, r2, asr #32
@ CHECK: add	r3, r1, r2, lsr #32     @ encoding: [0x22,0x30,0x81,0xe0]
@ CHECK: add	r3, r1, r2, asr #32     @ encoding: [0x42,0x30,0x81,0xe0]

@------------------------------------------------------------------------------
@ ADDS
@------------------------------------------------------------------------------
    adds r7, r8, #16711680
    adds r7, r8, $16711680
    adds r7, r8, 16711680
    adds r7, r8, #(0xff << 16)
    adds r7, r8, #-2147483638
    adds r7, r8, #42, #2
    adds r7, r8, #40, #2
    adds r7, r8, $40, $2
    adds r7, r8, 40, 2
    adds r7, r8, (2 * 20), (1 << 1)

@ CHECK: adds	r7, r8, #16711680         @ encoding: [0xff,0x78,0x98,0xe2]
@ CHECK: adds	r7, r8, #16711680         @ encoding: [0xff,0x78,0x98,0xe2]
@ CHECK: adds	r7, r8, #16711680         @ encoding: [0xff,0x78,0x98,0xe2]
@ CHECK: adds	r7, r8, #16711680         @ encoding: [0xff,0x78,0x98,0xe2]
@ CHECK: adds   r7, r8, #-2147483638      @ encoding: [0x2a,0x71,0x98,0xe2]
@ CHECK: adds   r7, r8, #-2147483638      @ encoding: [0x2a,0x71,0x98,0xe2]
@ CHECK: adds   r7, r8, #40, #2           @ encoding: [0x28,0x71,0x98,0xe2]
@ CHECK: adds   r7, r8, #40, #2           @ encoding: [0x28,0x71,0x98,0xe2]
@ CHECK: adds   r7, r8, #40, #2           @ encoding: [0x28,0x71,0x98,0xe2]
@ CHECK: adds   r7, r8, #40, #2           @ encoding: [0x28,0x71,0x98,0xe2]

@------------------------------------------------------------------------------
@ AND
@------------------------------------------------------------------------------
    and r10, r1, #0xf
    and r10, r1, $0xf
    and r10, r1, 0xf
    and r10, r1, -0xf
    and r7, r8, #(0xff << 16)
    and r7, r8, #-2147483638
    and r7, r8, #42, #2
    and r7, r8, #40, #2
    and r7, r8, $40, $2
    and r7, r8, 40, 2
    and r7, r8, (2 * 20), (1 << 1)
    and r10, r1, r6
    and r10, r1, r6, lsl #10
    and r10, r1, r6, lsr #10
    and r10, r1, r6, lsr #10
    and r10, r1, r6, asr #10
    and r10, r1, r6, ror #10
    and r6, r7, r8, lsl r2
    and r6, r7, r8, lsr r2
    and r6, r7, r8, asr r2
    and r6, r7, r8, ror r2
    and r10, r1, r6, rrx
    and r2, r3, #0x7fffffff
    and sp, sp, #0x7fffffff
    and pc, pc, #0x7fffffff

    @ destination register is optional
    and r1, #0xf
    and r1, $0xf
    and r1, 0xf
    and r1, -0xf
    and r7, #(0xff << 16)
    and r7, #-2147483638
    and r7, #42, #2
    and r7, #40, #2
    and r7, $40, $2
    and r7, 40, 2
    and r7, (2 * 20), (1 << 1)
    and r10, r1
    and r10, r1, lsl #10
    and r10, r1, lsr #10
    and r10, r1, lsr #10
    and r10, r1, asr #10
    and r10, r1, ror #10
    and r6, r7, lsl r2
    and r6, r7, lsr r2
    and r6, r7, asr r2
    and r6, r7, ror r2
    and r10, r1, rrx

@ CHECK: and	r10, r1, #15            @ encoding: [0x0f,0xa0,0x01,0xe2]
@ CHECK: and	r10, r1, #15            @ encoding: [0x0f,0xa0,0x01,0xe2]
@ CHECK: and	r10, r1, #15            @ encoding: [0x0f,0xa0,0x01,0xe2]
@ CHECK: bic	r10, r1, #14            @ encoding: [0x0e,0xa0,0xc1,0xe3]
@ CHECK: and	r7, r8, #16711680       @ encoding: [0xff,0x78,0x08,0xe2]
@ CHECK: and    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x08,0xe2]
@ CHECK: and    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x08,0xe2]
@ CHECK: and    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x08,0xe2]
@ CHECK: and    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x08,0xe2]
@ CHECK: and    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x08,0xe2]
@ CHECK: and    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x08,0xe2]
@ CHECK: and	r10, r1, r6             @ encoding: [0x06,0xa0,0x01,0xe0]
@ CHECK: and	r10, r1, r6, lsl #10    @ encoding: [0x06,0xa5,0x01,0xe0]
@ CHECK: and	r10, r1, r6, lsr #10    @ encoding: [0x26,0xa5,0x01,0xe0]
@ CHECK: and	r10, r1, r6, lsr #10    @ encoding: [0x26,0xa5,0x01,0xe0]
@ CHECK: and	r10, r1, r6, asr #10    @ encoding: [0x46,0xa5,0x01,0xe0]
@ CHECK: and	r10, r1, r6, ror #10    @ encoding: [0x66,0xa5,0x01,0xe0]
@ CHECK: and	r6, r7, r8, lsl r2      @ encoding: [0x18,0x62,0x07,0xe0]
@ CHECK: and	r6, r7, r8, lsr r2      @ encoding: [0x38,0x62,0x07,0xe0]
@ CHECK: and	r6, r7, r8, asr r2      @ encoding: [0x58,0x62,0x07,0xe0]
@ CHECK: and	r6, r7, r8, ror r2      @ encoding: [0x78,0x62,0x07,0xe0]
@ CHECK: and	r10, r1, r6, rrx        @ encoding: [0x66,0xa0,0x01,0xe0]
@ CHECK: bic	r2, r3, #-2147483648    @ encoding: [0x02,0x21,0xc3,0xe3]
@ CHECK: bic	sp, sp, #-2147483648    @ encoding: [0x02,0xd1,0xcd,0xe3]
@ CHECK: bic	pc, pc, #-2147483648    @ encoding: [0x02,0xf1,0xcf,0xe3]

@ CHECK: and	r1, r1, #15             @ encoding: [0x0f,0x10,0x01,0xe2]
@ CHECK: and	r1, r1, #15             @ encoding: [0x0f,0x10,0x01,0xe2]
@ CHECK: and	r1, r1, #15             @ encoding: [0x0f,0x10,0x01,0xe2]
@ CHECK: bic	r1, r1, #14             @ encoding: [0x0e,0x10,0xc1,0xe3]
@ CHECK: and	r7, r7, #16711680       @ encoding: [0xff,0x78,0x07,0xe2]
@ CHECK: and	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x07,0xe2]
@ CHECK: and	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x07,0xe2]
@ CHECK: and	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x07,0xe2]
@ CHECK: and	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x07,0xe2]
@ CHECK: and	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x07,0xe2]
@ CHECK: and	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x07,0xe2]
@ CHECK: and	r10, r10, r1            @ encoding: [0x01,0xa0,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, lsl #10   @ encoding: [0x01,0xa5,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, lsr #10   @ encoding: [0x21,0xa5,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, lsr #10   @ encoding: [0x21,0xa5,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, asr #10   @ encoding: [0x41,0xa5,0x0a,0xe0]
@ CHECK: and	r10, r10, r1, ror #10   @ encoding: [0x61,0xa5,0x0a,0xe0]
@ CHECK: and	r6, r6, r7, lsl r2      @ encoding: [0x17,0x62,0x06,0xe0]
@ CHECK: and	r6, r6, r7, lsr r2      @ encoding: [0x37,0x62,0x06,0xe0]
@ CHECK: and	r6, r6, r7, asr r2      @ encoding: [0x57,0x62,0x06,0xe0]
@ CHECK: and	r6, r6, r7, ror r2      @ encoding: [0x77,0x62,0x06,0xe0]
@ CHECK: and	r10, r10, r1, rrx       @ encoding: [0x61,0xa0,0x0a,0xe0]

    @ Test right shift by 32, which is encoded as 0
    and r3, r1, r2, lsr #32
    and r3, r1, r2, asr #32
@ CHECK: and	r3, r1, r2, lsr #32     @ encoding: [0x22,0x30,0x01,0xe0]
@ CHECK: and	r3, r1, r2, asr #32     @ encoding: [0x42,0x30,0x01,0xe0]

@------------------------------------------------------------------------------
@ ASR
@------------------------------------------------------------------------------
	asr r2, r4, #32
	asr r2, r4, #2
	asr r2, r4, #0
	asr r4, #2

@ CHECK: asr	r2, r4, #32             @ encoding: [0x44,0x20,0xa0,0xe1]
@ CHECK: asr	r2, r4, #2              @ encoding: [0x44,0x21,0xa0,0xe1]
@ CHECK: mov	r2, r4                  @ encoding: [0x04,0x20,0xa0,0xe1]
@ CHECK: asr	r4, r4, #2              @ encoding: [0x44,0x41,0xa0,0xe1]


@------------------------------------------------------------------------------
@ B
@------------------------------------------------------------------------------
        b _bar
        beq _baz

@ CHECK: b	_bar                    @ encoding: [A,A,A,0xea]
@ CHECK: @   fixup A - offset: 0, value: _bar, kind: fixup_arm_uncondbranch
@ CHECK-BE: b	_bar                    @ encoding: [0xea,A,A,A]
@ CHECK-BE: @   fixup A - offset: 0, value: _bar, kind: fixup_arm_uncondbranch
@ CHECK: beq	_baz                    @ encoding: [A,A,A,0x0a]
@ CHECK: @   fixup A - offset: 0, value: _baz, kind: fixup_arm_condbranch
@ CHECK-BE: beq	_baz                    @ encoding: [0x0a,A,A,A]
@ CHECK-BE: @   fixup A - offset: 0, value: _baz, kind: fixup_arm_condbranch


@------------------------------------------------------------------------------
@ BFC
@------------------------------------------------------------------------------
        bfc r5, #3, #17
        bfccc r5, #3, #17

@ CHECK: bfc	r5, #3, #17             @ encoding: [0x9f,0x51,0xd3,0xe7]
@ CHECK: bfclo	r5, #3, #17             @ encoding: [0x9f,0x51,0xd3,0x37]


@------------------------------------------------------------------------------
@ BFI
@------------------------------------------------------------------------------
        bfi r5, r2, #3, #17
        bfine r5, r2, #3, #17

@ CHECK: bfi	r5, r2, #3, #17         @ encoding: [0x92,0x51,0xd3,0xe7]
@ CHECK: bfine	r5, r2, #3, #17         @ encoding: [0x92,0x51,0xd3,0x17]


@------------------------------------------------------------------------------
@ BIC
@------------------------------------------------------------------------------
        bic r10, r1, #0xf
        bic r10, r1, $0xf
        bic r10, r1, 0xf
        bic r10, r1, -0xf
        bic r7, r8, #(0xff << 16)
        bic r7, r8, #-2147483638
        bic r7, r8, #42, #2
        bic r7, r8, #40, #2
        bic r7, r8, $40, $2
        bic r7, r8, 40, 2
        bic r7, r8, (2 * 20), (1 << 1)
        bic r10, r1, r6
        bic r10, r1, r6, lsl #10
        bic r10, r1, r6, lsr #10
        bic r10, r1, r6, lsr #10
        bic r10, r1, r6, asr #10
        bic r10, r1, r6, ror #10
        bic r6, r7, r8, lsl r2
        bic r6, r7, r8, lsr r2
        bic r6, r7, r8, asr r2
        bic r6, r7, r8, ror r2
        bic r10, r1, r6, rrx
        bic r2, r3, #0x7fffffff
        bic sp, sp, #0x7fffffff
        bic pc, pc, #0x7fffffff


        @ destination register is optional
        bic r1, #0xf
        bic r1, $0xf
        bic r1, 0xf
        bic r1, -0xf
        bic r7, #(0xff << 16)
        bic r7, #-2147483638
        bic r7, #42, #2
        bic r7, #40, #2
        bic r7, $40, $2
        bic r7, 40, 2
        bic r7, (2 * 20), (1 << 1)
        bic r10, r1
        bic r10, r1, lsl #10
        bic r10, r1, lsr #10
        bic r10, r1, lsr #10
        bic r10, r1, asr #10
        bic r10, r1, ror #10
        bic r6, r7, lsl r2
        bic r6, r7, lsr r2
        bic r6, r7, asr r2
        bic r6, r7, ror r2
        bic r10, r1, rrx

@ CHECK: bic	r10, r1, #15            @ encoding: [0x0f,0xa0,0xc1,0xe3]
@ CHECK: bic	r10, r1, #15            @ encoding: [0x0f,0xa0,0xc1,0xe3]
@ CHECK: bic	r10, r1, #15            @ encoding: [0x0f,0xa0,0xc1,0xe3]
@ CHECK: and	r10, r1, #14            @ encoding: [0x0e,0xa0,0x01,0xe2]
@ CHECK: bic	r7, r8, #16711680       @ encoding: [0xff,0x78,0xc8,0xe3]
@ CHECK: bic    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0xc8,0xe3]
@ CHECK: bic    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0xc8,0xe3]
@ CHECK: bic    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xc8,0xe3]
@ CHECK: bic    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xc8,0xe3]
@ CHECK: bic    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xc8,0xe3]
@ CHECK: bic	r10, r1, r6             @ encoding: [0x06,0xa0,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, lsl #10    @ encoding: [0x06,0xa5,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, lsr #10    @ encoding: [0x26,0xa5,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, lsr #10    @ encoding: [0x26,0xa5,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, asr #10    @ encoding: [0x46,0xa5,0xc1,0xe1]
@ CHECK: bic	r10, r1, r6, ror #10    @ encoding: [0x66,0xa5,0xc1,0xe1]
@ CHECK: bic	r6, r7, r8, lsl r2      @ encoding: [0x18,0x62,0xc7,0xe1]
@ CHECK: bic	r6, r7, r8, lsr r2      @ encoding: [0x38,0x62,0xc7,0xe1]
@ CHECK: bic	r6, r7, r8, asr r2      @ encoding: [0x58,0x62,0xc7,0xe1]
@ CHECK: bic	r6, r7, r8, ror r2      @ encoding: [0x78,0x62,0xc7,0xe1]
@ CHECK: bic	r10, r1, r6, rrx        @ encoding: [0x66,0xa0,0xc1,0xe1]
@ CHECK: and  r2, r3, #-2147483648    @ encoding: [0x02,0x21,0x03,0xe2]
@ CHECK: and  sp, sp, #-2147483648    @ encoding: [0x02,0xd1,0x0d,0xe2]
@ CHECK: and  pc, pc, #-2147483648    @ encoding: [0x02,0xf1,0x0f,0xe2]


@ CHECK: bic	r1, r1, #15             @ encoding: [0x0f,0x10,0xc1,0xe3]
@ CHECK: bic	r1, r1, #15             @ encoding: [0x0f,0x10,0xc1,0xe3]
@ CHECK: bic	r1, r1, #15             @ encoding: [0x0f,0x10,0xc1,0xe3]
@ CHECK: and	r1, r1, #14             @ encoding: [0x0e,0x10,0x01,0xe2]
@ CHECK: bic	r7, r7, #16711680       @ encoding: [0xff,0x78,0xc7,0xe3]
@ CHECK: bic    r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0xc7,0xe3]
@ CHECK: bic    r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0xc7,0xe3]
@ CHECK: bic    r7, r7, #40, #2         @ encoding: [0x28,0x71,0xc7,0xe3]
@ CHECK: bic    r7, r7, #40, #2         @ encoding: [0x28,0x71,0xc7,0xe3]
@ CHECK: bic    r7, r7, #40, #2         @ encoding: [0x28,0x71,0xc7,0xe3]
@ CHECK: bic    r7, r7, #40, #2         @ encoding: [0x28,0x71,0xc7,0xe3]
@ CHECK: bic	r10, r10, r1            @ encoding: [0x01,0xa0,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, lsl #10   @ encoding: [0x01,0xa5,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, lsr #10   @ encoding: [0x21,0xa5,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, lsr #10   @ encoding: [0x21,0xa5,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, asr #10   @ encoding: [0x41,0xa5,0xca,0xe1]
@ CHECK: bic	r10, r10, r1, ror #10   @ encoding: [0x61,0xa5,0xca,0xe1]
@ CHECK: bic	r6, r6, r7, lsl r2      @ encoding: [0x17,0x62,0xc6,0xe1]
@ CHECK: bic	r6, r6, r7, lsr r2      @ encoding: [0x37,0x62,0xc6,0xe1]
@ CHECK: bic	r6, r6, r7, asr r2      @ encoding: [0x57,0x62,0xc6,0xe1]
@ CHECK: bic	r6, r6, r7, ror r2      @ encoding: [0x77,0x62,0xc6,0xe1]
@ CHECK: bic	r10, r10, r1, rrx       @ encoding: [0x61,0xa0,0xca,0xe1]

    @ Test right shift by 32, which is encoded as 0
    bic r3, r1, r2, lsr #32
    bic r3, r1, r2, asr #32
@ CHECK: bic	r3, r1, r2, lsr #32     @ encoding: [0x22,0x30,0xc1,0xe1]
@ CHECK: bic	r3, r1, r2, asr #32     @ encoding: [0x42,0x30,0xc1,0xe1]

@------------------------------------------------------------------------------
@ BKPT
@------------------------------------------------------------------------------
        bkpt #10
        bkpt #65535

@ CHECK: bkpt  #10                      @ encoding: [0x7a,0x00,0x20,0xe1]
@ CHECK: bkpt  #65535                   @ encoding: [0x7f,0xff,0x2f,0xe1]

@------------------------------------------------------------------------------
@ BL/BLX (immediate)
@------------------------------------------------------------------------------

        bl _bar
        bleq _bar
        blx _bar
        blls #28634268
        blx	#32424576
        blx	#16212288

@ CHECK: bl  _bar @ encoding: [A,A,A,0xeb]
@ CHECK:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_uncondbl
@ CHECK-BE: bl  _bar @ encoding: [0xeb,A,A,A]
@ CHECK-BE:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_uncondbl
@ CHECK: bleq  _bar @ encoding: [A,A,A,0x0b]
@ CHECK:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_condbl
@ CHECK-BE: bleq  _bar @ encoding: [0x0b,A,A,A]
@ CHECK-BE:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_condbl
@ CHECK: blx	_bar @ encoding: [A,A,A,0xfa]
@ CHECK:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_blx
@ CHECK-BE: blx	_bar @ encoding: [0xfa,A,A,A]
@ CHECK-BE:   @   fixup A - offset: 0, value: _bar, kind: fixup_arm_blx
@ CHECK: blls	#28634268               @ encoding: [0x27,0x3b,0x6d,0x9b]
@ CHECK: blx	#32424576               @ encoding: [0xa0,0xb0,0x7b,0xfa]
@ CHECK: blx	#16212288               @ encoding: [0x50,0xd8,0x3d,0xfa]
@------------------------------------------------------------------------------
@ BLX (register)
@------------------------------------------------------------------------------
        blx r2
        blxne r2

@ CHECK: blx r2                         @ encoding: [0x32,0xff,0x2f,0xe1]
@ CHECK: blxne r2                       @ encoding: [0x32,0xff,0x2f,0x11]

@------------------------------------------------------------------------------
@ BX
@------------------------------------------------------------------------------
        bx r2
        bxne r2

@ CHECK: bx	r2                      @ encoding: [0x12,0xff,0x2f,0xe1]
@ CHECK: bxne	r2                      @ encoding: [0x12,0xff,0x2f,0x11]

@------------------------------------------------------------------------------
@ BXJ
@------------------------------------------------------------------------------
        bxj r2
        bxjne r2

@ CHECK: bxj	r2                      @ encoding: [0x22,0xff,0x2f,0xe1]
@ CHECK: bxjne	r2                      @ encoding: [0x22,0xff,0x2f,0x11]


@------------------------------------------------------------------------------
@ CDP/CDP2
@------------------------------------------------------------------------------
        cdp  p7, #1, c1, c1, c1, #4
        cdp2  p7, #1, c1, c1, c1, #4
        cdp2   p12, #0, c6, c12, c0, #7

@ CHECK: cdp  p7, #1, c1, c1, c1, #4     @ encoding: [0x81,0x17,0x11,0xee]
@ CHECK: cdp2  p7, #1, c1, c1, c1, #4    @ encoding: [0x81,0x17,0x11,0xfe]
@ CHECK: cdp2  p12, #0, c6, c12, c0, #7   @ encoding: [0xe0,0x6c,0x0c,0xfe]

        cdpne  p7, #1, c1, c1, c1, #4
@ CHECK: cdpne  p7, #1, c1, c1, c1, #4     @ encoding: [0x81,0x17,0x11,0x1e]

@------------------------------------------------------------------------------
@ CLREX
@------------------------------------------------------------------------------
        clrex

@ CHECK: clrex                           @ encoding: [0x1f,0xf0,0x7f,0xf5]


@------------------------------------------------------------------------------
@ CLZ
@------------------------------------------------------------------------------
    clz r1, r2
    clzeq r1, r2

@ CHECK: clz r1, r2                      @ encoding: [0x12,0x1f,0x6f,0xe1]
@ CHECK: clzeq r1, r2                    @ encoding: [0x12,0x1f,0x6f,0x01]

@------------------------------------------------------------------------------
@ CMN
@------------------------------------------------------------------------------
        cmn r1, #0xf
        cmn r1, $0xf
        cmn r1, 0xf
        cmn r1, -0xf
        cmn r7, #(0xff << 16)
        cmn r7, #-2147483638
        cmn r7, #42, #2
        cmn r7, #40, #2
        cmn r7, $40, $2
        cmn r7, 40, 2
        cmn r7, (20 * 2), (1 << 1)
        cmn r1, r6
        cmn r1, r6, lsl #10
        cmn r1, r6, lsr #10
        cmn sp, r6, lsr #10
        cmn r1, r6, asr #10
        cmn r1, r6, ror #10
        cmn r7, r8, lsl r2
        cmn sp, r8, lsr r2
        cmn r7, r8, asr r2
        cmn r7, r8, ror r2
        cmn r1, r6, rrx

@ CHECK: cmn	r1, #15                 @ encoding: [0x0f,0x00,0x71,0xe3]
@ CHECK: cmn	r1, #15                 @ encoding: [0x0f,0x00,0x71,0xe3]
@ CHECK: cmn	r1, #15                 @ encoding: [0x0f,0x00,0x71,0xe3]
@ CHECK: cmp	r1, #15                 @ encoding: [0x0f,0x00,0x51,0xe3]
@ CHECK: cmn	r7, #16711680           @ encoding: [0xff,0x08,0x77,0xe3]
@ CHECK: cmn	r7, #-2147483638        @ encoding: [0x2a,0x01,0x77,0xe3]
@ CHECK: cmn	r7, #-2147483638        @ encoding: [0x2a,0x01,0x77,0xe3]
@ CHECK: cmn	r7, #40, #2             @ encoding: [0x28,0x01,0x77,0xe3]
@ CHECK: cmn	r7, #40, #2             @ encoding: [0x28,0x01,0x77,0xe3]
@ CHECK: cmn	r7, #40, #2             @ encoding: [0x28,0x01,0x77,0xe3]
@ CHECK: cmn	r7, #40, #2             @ encoding: [0x28,0x01,0x77,0xe3]
@ CHECK: cmn	r1, r6                  @ encoding: [0x06,0x00,0x71,0xe1]
@ CHECK: cmn	r1, r6, lsl #10         @ encoding: [0x06,0x05,0x71,0xe1]
@ CHECK: cmn	r1, r6, lsr #10         @ encoding: [0x26,0x05,0x71,0xe1]
@ CHECK: cmn	sp, r6, lsr #10         @ encoding: [0x26,0x05,0x7d,0xe1]
@ CHECK: cmn	r1, r6, asr #10         @ encoding: [0x46,0x05,0x71,0xe1]
@ CHECK: cmn	r1, r6, ror #10         @ encoding: [0x66,0x05,0x71,0xe1]
@ CHECK: cmn	r7, r8, lsl r2          @ encoding: [0x18,0x02,0x77,0xe1]
@ CHECK: cmn	sp, r8, lsr r2          @ encoding: [0x38,0x02,0x7d,0xe1]
@ CHECK: cmn	r7, r8, asr r2          @ encoding: [0x58,0x02,0x77,0xe1]
@ CHECK: cmn	r7, r8, ror r2          @ encoding: [0x78,0x02,0x77,0xe1]
@ CHECK: cmn	r1, r6, rrx             @ encoding: [0x66,0x00,0x71,0xe1]

@------------------------------------------------------------------------------
@ CMP
@------------------------------------------------------------------------------
        cmp r1, #0xf
        cmp r1, $0xf
        cmp r1, 0xf
        cmp r1, -0xf
        cmp r7, #(0xff << 16)
        cmp r7, #-2147483638
        cmp r7, #42, #2
        cmp r7, #40, #2
        cmp r7, $40, $2
        cmp r7, 40, 2
        cmp r7, (2 * 20), (1 << 1)
        cmp r1, r6
        cmp r1, r6, lsl #10
        cmp r1, r6, lsr #10
        cmp sp, r6, lsr #10
        cmp r1, r6, asr #10
        cmp r1, r6, ror #10
        cmp r7, r8, lsl r2
        cmp sp, r8, lsr r2
        cmp r7, r8, asr r2
        cmp r7, r8, ror r2
        cmp r1, r6, rrx
        cmp r0, #-2
        cmp lr, #0

@ CHECK: cmp	r1, #15                 @ encoding: [0x0f,0x00,0x51,0xe3]
@ CHECK: cmp	r1, #15                 @ encoding: [0x0f,0x00,0x51,0xe3]
@ CHECK: cmp	r1, #15                 @ encoding: [0x0f,0x00,0x51,0xe3]
@ CHECK: cmn	r1, #15                 @ encoding: [0x0f,0x00,0x71,0xe3]
@ CHECK: cmp	r7, #16711680           @ encoding: [0xff,0x08,0x57,0xe3]
@ CHECK: cmp	r7, #-2147483638        @ encoding: [0x2a,0x01,0x57,0xe3]
@ CHECK: cmp	r7, #-2147483638        @ encoding: [0x2a,0x01,0x57,0xe3]
@ CHECK: cmp    r7, #40, #2             @ encoding: [0x28,0x01,0x57,0xe3]
@ CHECK: cmp    r7, #40, #2             @ encoding: [0x28,0x01,0x57,0xe3]
@ CHECK: cmp    r7, #40, #2             @ encoding: [0x28,0x01,0x57,0xe3]
@ CHECK: cmp    r7, #40, #2             @ encoding: [0x28,0x01,0x57,0xe3]
@ CHECK: cmp	r1, r6                  @ encoding: [0x06,0x00,0x51,0xe1]
@ CHECK: cmp	r1, r6, lsl #10         @ encoding: [0x06,0x05,0x51,0xe1]
@ CHECK: cmp	r1, r6, lsr #10         @ encoding: [0x26,0x05,0x51,0xe1]
@ CHECK: cmp	sp, r6, lsr #10         @ encoding: [0x26,0x05,0x5d,0xe1]
@ CHECK: cmp	r1, r6, asr #10         @ encoding: [0x46,0x05,0x51,0xe1]
@ CHECK: cmp	r1, r6, ror #10         @ encoding: [0x66,0x05,0x51,0xe1]
@ CHECK: cmp	r7, r8, lsl r2          @ encoding: [0x18,0x02,0x57,0xe1]
@ CHECK: cmp	sp, r8, lsr r2          @ encoding: [0x38,0x02,0x5d,0xe1]
@ CHECK: cmp	r7, r8, asr r2          @ encoding: [0x58,0x02,0x57,0xe1]
@ CHECK: cmp	r7, r8, ror r2          @ encoding: [0x78,0x02,0x57,0xe1]
@ CHECK: cmp	r1, r6, rrx             @ encoding: [0x66,0x00,0x51,0xe1]
@ CHECK: cmn	r0, #2                  @ encoding: [0x02,0x00,0x70,0xe3]
@ CHECK: cmp  lr, #0                    @ encoding: [0x00,0x00,0x5e,0xe3]


@------------------------------------------------------------------------------
@ CPS
@------------------------------------------------------------------------------
        cpsie  aif
        cpsie  AIF
        cps  #15
        cpsid  if, #10

@ CHECK: cpsie  aif @ encoding: [0xc0,0x01,0x08,0xf1]
@ CHECK: cpsie  aif @ encoding: [0xc0,0x01,0x08,0xf1]
@ CHECK: cps  #15 @ encoding: [0x0f,0x00,0x02,0xf1]
@ CHECK: cpsid  if, #10 @ encoding: [0xca,0x00,0x0e,0xf1]


@------------------------------------------------------------------------------
@ DBG
@------------------------------------------------------------------------------
        dbg #0
        dbg #5
        dbg #15

@ CHECK: dbg #0                         @ encoding: [0xf0,0xf0,0x20,0xe3]
@ CHECK: dbg #5                         @ encoding: [0xf5,0xf0,0x20,0xe3]
@ CHECK: dbg #15                        @ encoding: [0xff,0xf0,0x20,0xe3]


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

@ CHECK: dmb	sy                      @ encoding: [0x5f,0xf0,0x7f,0xf5]
@ CHECK: dmb	st                      @ encoding: [0x5e,0xf0,0x7f,0xf5]
@ CHECK: dmb	#0xd                    @ encoding: [0x5d,0xf0,0x7f,0xf5]
@ CHECK: dmb	#0xc                    @ encoding: [0x5c,0xf0,0x7f,0xf5]
@ CHECK: dmb	ish                     @ encoding: [0x5b,0xf0,0x7f,0xf5]
@ CHECK: dmb	ishst                   @ encoding: [0x5a,0xf0,0x7f,0xf5]
@ CHECK: dmb	#0x9                    @ encoding: [0x59,0xf0,0x7f,0xf5]
@ CHECK: dmb	#0x8                    @ encoding: [0x58,0xf0,0x7f,0xf5]
@ CHECK: dmb	nsh                     @ encoding: [0x57,0xf0,0x7f,0xf5]
@ CHECK: dmb	nshst                   @ encoding: [0x56,0xf0,0x7f,0xf5]
@ CHECK: dmb	#0x5                    @ encoding: [0x55,0xf0,0x7f,0xf5]
@ CHECK: dmb	#0x4                    @ encoding: [0x54,0xf0,0x7f,0xf5]
@ CHECK: dmb	osh                     @ encoding: [0x53,0xf0,0x7f,0xf5]
@ CHECK: dmb	oshst                   @ encoding: [0x52,0xf0,0x7f,0xf5]
@ CHECK: dmb	#0x1                    @ encoding: [0x51,0xf0,0x7f,0xf5]
@ CHECK: dmb	#0x0                    @ encoding: [0x50,0xf0,0x7f,0xf5]

@ CHECK: dmb	sy                      @ encoding: [0x5f,0xf0,0x7f,0xf5]
@ CHECK: dmb	st                      @ encoding: [0x5e,0xf0,0x7f,0xf5]
@ CHECK: dmb	ish                     @ encoding: [0x5b,0xf0,0x7f,0xf5]
@ CHECK: dmb	ish                     @ encoding: [0x5b,0xf0,0x7f,0xf5]
@ CHECK: dmb	ishst                   @ encoding: [0x5a,0xf0,0x7f,0xf5]
@ CHECK: dmb	ishst                   @ encoding: [0x5a,0xf0,0x7f,0xf5]
@ CHECK: dmb	nsh                     @ encoding: [0x57,0xf0,0x7f,0xf5]
@ CHECK: dmb	nsh                     @ encoding: [0x57,0xf0,0x7f,0xf5]
@ CHECK: dmb	nshst                   @ encoding: [0x56,0xf0,0x7f,0xf5]
@ CHECK: dmb	nshst                   @ encoding: [0x56,0xf0,0x7f,0xf5]
@ CHECK: dmb	osh                     @ encoding: [0x53,0xf0,0x7f,0xf5]
@ CHECK: dmb	oshst                   @ encoding: [0x52,0xf0,0x7f,0xf5]
@ CHECK: dmb	sy                      @ encoding: [0x5f,0xf0,0x7f,0xf5]

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

        dsb 8
        dsb 7

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

@ CHECK: dsb	sy                      @ encoding: [0x4f,0xf0,0x7f,0xf5]
@ CHECK: dsb	st                      @ encoding: [0x4e,0xf0,0x7f,0xf5]
@ CHECK: dsb	#0xd                    @ encoding: [0x4d,0xf0,0x7f,0xf5]
@ CHECK: dsb	#0xc                    @ encoding: [0x4c,0xf0,0x7f,0xf5]
@ CHECK: dsb	ish                     @ encoding: [0x4b,0xf0,0x7f,0xf5]
@ CHECK: dsb	ishst                   @ encoding: [0x4a,0xf0,0x7f,0xf5]
@ CHECK: dsb	#0x9                    @ encoding: [0x49,0xf0,0x7f,0xf5]
@ CHECK: dsb	#0x8                    @ encoding: [0x48,0xf0,0x7f,0xf5]
@ CHECK: dsb	nsh                     @ encoding: [0x47,0xf0,0x7f,0xf5]
@ CHECK: dsb	nshst                   @ encoding: [0x46,0xf0,0x7f,0xf5]
@ CHECK: dsb	#0x5                    @ encoding: [0x45,0xf0,0x7f,0xf5]
@ CHECK: pssbb                          @ encoding: [0x44,0xf0,0x7f,0xf5]
@ CHECK: dsb	osh                     @ encoding: [0x43,0xf0,0x7f,0xf5]
@ CHECK: dsb	oshst                   @ encoding: [0x42,0xf0,0x7f,0xf5]
@ CHECK: dsb	#0x1                    @ encoding: [0x41,0xf0,0x7f,0xf5]
@ CHECK: ssbb                           @ encoding: [0x40,0xf0,0x7f,0xf5]

@ CHECK: dsb	#0x8                    @ encoding: [0x48,0xf0,0x7f,0xf5]
@ CHECK: dsb	nsh                     @ encoding: [0x47,0xf0,0x7f,0xf5]

@ CHECK: dsb	sy                      @ encoding: [0x4f,0xf0,0x7f,0xf5]
@ CHECK: dsb	st                      @ encoding: [0x4e,0xf0,0x7f,0xf5]
@ CHECK: dsb	ish                     @ encoding: [0x4b,0xf0,0x7f,0xf5]
@ CHECK: dsb	ish                     @ encoding: [0x4b,0xf0,0x7f,0xf5]
@ CHECK: dsb	ishst                   @ encoding: [0x4a,0xf0,0x7f,0xf5]
@ CHECK: dsb	ishst                   @ encoding: [0x4a,0xf0,0x7f,0xf5]
@ CHECK: dsb	nsh                     @ encoding: [0x47,0xf0,0x7f,0xf5]
@ CHECK: dsb	nsh                     @ encoding: [0x47,0xf0,0x7f,0xf5]
@ CHECK: dsb	nshst                   @ encoding: [0x46,0xf0,0x7f,0xf5]
@ CHECK: dsb	nshst                   @ encoding: [0x46,0xf0,0x7f,0xf5]
@ CHECK: dsb	osh                     @ encoding: [0x43,0xf0,0x7f,0xf5]
@ CHECK: dsb	oshst                   @ encoding: [0x42,0xf0,0x7f,0xf5]
@ CHECK: dsb	sy                      @ encoding: [0x4f,0xf0,0x7f,0xf5]

@ With capitals
        dsb SY
        dsb OSHST

@ CHECK: dsb	sy                      @ encoding: [0x4f,0xf0,0x7f,0xf5]
@ CHECK: dsb	oshst                   @ encoding: [0x42,0xf0,0x7f,0xf5]
@------------------------------------------------------------------------------
@ EOR
@------------------------------------------------------------------------------
        eor r4, r5, #0xf000
        eor r4, r5, $0xf000
        eor r4, r5, 0xf000
        eor r7, r8, #(0xff << 16)
        eor r7, r8, #-2147483638
        eor r7, r8, #42, #2
        eor r7, r8, #40, #2
        eor r7, r8, $40, $2
        eor r7, r8, 40, 2
        eor r7, r8, (20 * 2), (1 << 1)
        eor r4, r5, r6
        eor r4, r5, r6, lsl #5
        eor r4, r5, r6, lsr #5
        eor r4, r5, r6, lsr #5
        eor r4, r5, r6, asr #5
        eor r4, r5, r6, ror #5
        eor r6, r7, r8, lsl r9
        eor r6, r7, r8, lsr r9
        eor r6, r7, r8, asr r9
        eor r6, r7, r8, ror r9
        eor r4, r5, r6, rrx

        @ destination register is optional
        eor r5, #0xf000
        eor r5, $0xf000
        eor r5, 0xf000
        eor r7, #(0xff << 16)
        eor r7, #-2147483638
        eor r7, #42, #2
        eor r7, #40, #2
        eor r7, $40, $2
        eor r7, 40, 2
        eor r7, (20 * 2), (1 << 1)
        eor r4, r5
        eor r4, r5, lsl #5
        eor r4, r5, lsr #5
        eor r4, r5, lsr #5
        eor r4, r5, asr #5
        eor r4, r5, ror #5
        eor r6, r7, lsl r9
        eor r6, r7, lsr r9
        eor r6, r7, asr r9
        eor r6, r7, ror r9
        eor r4, r5, rrx

@ CHECK: eor	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x25,0xe2]
@ CHECK: eor	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x25,0xe2]
@ CHECK: eor	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x25,0xe2]
@ CHECK: eor	r7, r8, #16711680       @ encoding: [0xff,0x78,0x28,0xe2]
@ CHECK: eor    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x28,0xe2]
@ CHECK: eor    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x28,0xe2]
@ CHECK: eor    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x28,0xe2]
@ CHECK: eor    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x28,0xe2]
@ CHECK: eor    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x28,0xe2]
@ CHECK: eor    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x28,0xe2]
@ CHECK: eor	r4, r5, r6              @ encoding: [0x06,0x40,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x25,0xe0]
@ CHECK: eor	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x25,0xe0]
@ CHECK: eor	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x27,0xe0]
@ CHECK: eor	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x27,0xe0]
@ CHECK: eor	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x27,0xe0]
@ CHECK: eor	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x27,0xe0]
@ CHECK: eor	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x25,0xe0]


@ CHECK: eor	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x25,0xe2]
@ CHECK: eor	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x25,0xe2]
@ CHECK: eor	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x25,0xe2]
@ CHECK: eor	r7, r7, #16711680       @ encoding: [0xff,0x78,0x27,0xe2]
@ CHECK: eor	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x27,0xe2]
@ CHECK: eor	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x27,0xe2]
@ CHECK: eor	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x27,0xe2]
@ CHECK: eor	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x27,0xe2]
@ CHECK: eor	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x27,0xe2]
@ CHECK: eor	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x27,0xe2]
@ CHECK: eor	r4, r4, r5              @ encoding: [0x05,0x40,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x24,0xe0]
@ CHECK: eor	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x24,0xe0]
@ CHECK: eor	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x26,0xe0]
@ CHECK: eor	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x26,0xe0]
@ CHECK: eor	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x26,0xe0]
@ CHECK: eor	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x26,0xe0]
@ CHECK: eor	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x24,0xe0]

    @ Test right shift by 32, which is encoded as 0
    eor r3, r1, r2, lsr #32
    eor r3, r1, r2, asr #32
@ CHECK: eor	r3, r1, r2, lsr #32     @ encoding: [0x22,0x30,0x21,0xe0]
@ CHECK: eor	r3, r1, r2, asr #32     @ encoding: [0x42,0x30,0x21,0xe0]

@------------------------------------------------------------------------------
@ ISB
@------------------------------------------------------------------------------
        isb sy
        isb
        isb #15
        isb #1

@ CHECK: isb sy                         @ encoding: [0x6f,0xf0,0x7f,0xf5]
@ CHECK: isb sy                         @ encoding: [0x6f,0xf0,0x7f,0xf5]
@ CHECK: isb sy                         @ encoding: [0x6f,0xf0,0x7f,0xf5]
@ CHECK: isb #0x1                       @ encoding: [0x61,0xf0,0x7f,0xf5]


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

        ldclo p12, c4, [r0, #4]
        ldchi p13, c5, [r1]
        ldccs p14, c6, [r2, #-224]
        ldccc p15, c7, [r3, #-120]!
        ldceq p5, c8, [r4], #16
        ldcgt p4, c9, [r5], #-72
        ldcllt p3, c10, [r6, #4]
        ldclge p2, c11, [r7]
        ldclle p1, c12, [r8, #-224]
        ldclne p0, c13, [r9, #-120]!
        ldcleq p6, c14, [r10], #16
        ldclhi p7, c15, [r11], #-72

        ldc2 p2, c8, [r1], { 25 }

@ CHECK: ldc2	p0, c8, [r1, #4]        @ encoding: [0x01,0x80,0x91,0xfd]
@ CHECK: ldc2	p1, c7, [r2]            @ encoding: [0x00,0x71,0x92,0xfd]
@ CHECK: ldc2	p2, c6, [r3, #-224]     @ encoding: [0x38,0x62,0x13,0xfd]
@ CHECK: ldc2	p3, c5, [r4, #-120]!    @ encoding: [0x1e,0x53,0x34,0xfd]
@ CHECK: ldc2	p4, c4, [r5], #16       @ encoding: [0x04,0x44,0xb5,0xfc]
@ CHECK: ldc2	p5, c3, [r6], #-72      @ encoding: [0x12,0x35,0x36,0xfc]
@ CHECK: ldc2l	p6, c2, [r7, #4]        @ encoding: [0x01,0x26,0xd7,0xfd]
@ CHECK: ldc2l	p7, c1, [r8]            @ encoding: [0x00,0x17,0xd8,0xfd]
@ CHECK: ldc2l	p8, c0, [r9, #-224]     @ encoding: [0x38,0x08,0x59,0xfd]
@ CHECK: ldc2l	p9, c1, [r10, #-120]!   @ encoding: [0x1e,0x19,0x7a,0xfd]
@ CHECK: ldc2l	p0, c2, [r11], #16      @ encoding: [0x04,0x20,0xfb,0xfc]
@ CHECK: ldc2l	p1, c3, [r12], #-72     @ encoding: [0x12,0x31,0x7c,0xfc]

@ CHECK: ldc	p12, c4, [r0, #4]       @ encoding: [0x01,0x4c,0x90,0xed]
@ CHECK: ldc	p13, c5, [r1]           @ encoding: [0x00,0x5d,0x91,0xed]
@ CHECK: ldc	p14, c6, [r2, #-224]    @ encoding: [0x38,0x6e,0x12,0xed]
@ CHECK: ldc	p15, c7, [r3, #-120]!   @ encoding: [0x1e,0x7f,0x33,0xed]
@ CHECK: ldc	p5, c8, [r4], #16       @ encoding: [0x04,0x85,0xb4,0xec]
@ CHECK: ldc	p4, c9, [r5], #-72      @ encoding: [0x12,0x94,0x35,0xec]
@ CHECK: ldcl	p3, c10, [r6, #4]       @ encoding: [0x01,0xa3,0xd6,0xed]
@ CHECK: ldcl	p2, c11, [r7]           @ encoding: [0x00,0xb2,0xd7,0xed]
@ CHECK: ldcl	p1, c12, [r8, #-224]    @ encoding: [0x38,0xc1,0x58,0xed]
@ CHECK: ldcl	p0, c13, [r9, #-120]!   @ encoding: [0x1e,0xd0,0x79,0xed]
@ CHECK: ldcl	p6, c14, [r10], #16     @ encoding: [0x04,0xe6,0xfa,0xec]
@ CHECK: ldcl	p7, c15, [r11], #-72    @ encoding: [0x12,0xf7,0x7b,0xec]

@ CHECK: ldclo	p12, c4, [r0, #4]       @ encoding: [0x01,0x4c,0x90,0x3d]
@ CHECK: ldchi	p13, c5, [r1]           @ encoding: [0x00,0x5d,0x91,0x8d]
@ CHECK: ldchs	p14, c6, [r2, #-224]    @ encoding: [0x38,0x6e,0x12,0x2d]
@ CHECK: ldclo	p15, c7, [r3, #-120]!   @ encoding: [0x1e,0x7f,0x33,0x3d]
@ CHECK: ldceq	p5, c8, [r4], #16       @ encoding: [0x04,0x85,0xb4,0x0c]
@ CHECK: ldcgt	p4, c9, [r5], #-72      @ encoding: [0x12,0x94,0x35,0xcc]
@ CHECK: ldcllt	p3, c10, [r6, #4]       @ encoding: [0x01,0xa3,0xd6,0xbd]
@ CHECK: ldclge	p2, c11, [r7]           @ encoding: [0x00,0xb2,0xd7,0xad]
@ CHECK: ldclle	p1, c12, [r8, #-224]    @ encoding: [0x38,0xc1,0x58,0xdd]
@ CHECK: ldclne	p0, c13, [r9, #-120]!   @ encoding: [0x1e,0xd0,0x79,0x1d]
@ CHECK: ldcleq	p6, c14, [r10], #16     @ encoding: [0x04,0xe6,0xfa,0x0c]
@ CHECK: ldclhi	p7, c15, [r11], #-72    @ encoding: [0x12,0xf7,0x7b,0x8c]

@ CHECK: ldc2	p2, c8, [r1], {25}      @ encoding: [0x19,0x82,0x91,0xfc]


@------------------------------------------------------------------------------
@ LDM*
@------------------------------------------------------------------------------
        ldm       r2, {r1,r3-r6,sp}
        ldmia     r2, {r1,r3-r6,sp}
        ldmib     r2, {r1,r3-r6,sp}
        ldmda     r2, {r1,r3-r6,sp}
        ldmdb     r2, {r1,r3-r6,sp}
        ldmfd     r2, {r1,r3-r6,sp}

        @ with update
        ldm       r2!, {r1,r3-r6,sp}
        ldmib     r2!, {r1,r3-r6,sp}
        ldmda     r2!, {r1,r3-r6,sp}
        ldmdb     r2!, {r1,r3-r6,sp}

        @ system version
        ldm r0, {r0, r2, lr}^
        ldm sp!, {r0-r3, pc}^

@ CHECK: ldm   r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe8]
@ CHECK: ldm   r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe8]
@ CHECK: ldmib r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe9]
@ CHECK: ldmda r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x12,0xe8]
@ CHECK: ldmdb r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x12,0xe9]
@ CHECK: ldm   r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x92,0xe8]

@ CHECK: ldm   r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xb2,0xe8]
@ CHECK: ldmib r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xb2,0xe9]
@ CHECK: ldmda r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x32,0xe8]
@ CHECK: ldmdb r2!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x32,0xe9]
@ CHECK: ldm	r0, {r0, r2, lr} ^          @ encoding: [0x05,0x40,0xd0,0xe8]
@ CHECK: ldm	sp!, {r0, r1, r2, r3, pc} ^ @ encoding: [0x0f,0x80,0xfd,0xe8]


@------------------------------------------------------------------------------
@ LDREX/LDREXB/LDREXH/LDREXD
@------------------------------------------------------------------------------
        ldrexb  r3, [r4]
        ldrexh  r2, [r5]
        ldrex  r1, [r7]
        ldrexd  r6, r7, [r8]

@ CHECK: ldrexb	r3, [r4]                @ encoding: [0x9f,0x3f,0xd4,0xe1]
@ CHECK: ldrexh	r2, [r5]                @ encoding: [0x9f,0x2f,0xf5,0xe1]
@ CHECK: ldrex	r1, [r7]                @ encoding: [0x9f,0x1f,0x97,0xe1]
@ CHECK: ldrexd	r6, r7, [r8]            @ encoding: [0x9f,0x6f,0xb8,0xe1]

@------------------------------------------------------------------------------
@ LDRHT
@------------------------------------------------------------------------------
        ldrhthi	r8, [r11], #-0
        ldrhthi	r8, [r11], #0

@ CHECK: ldrhthi r8, [r11], #-0         @ encoding: [0xb0,0x80,0x7b,0x80]
@ CHECK: ldrhthi r8, [r11], #0          @ encoding: [0xb0,0x80,0xfb,0x80]

@------------------------------------------------------------------------------
@ LSL
@------------------------------------------------------------------------------
	lsl r2, r4, #31
	lsl r2, r4, #1
	lsl r2, r4, #0
	lsl r4, #1

@ CHECK: lsl	r2, r4, #31             @ encoding: [0x84,0x2f,0xa0,0xe1]
@ CHECK: lsl	r2, r4, #1              @ encoding: [0x84,0x20,0xa0,0xe1]
@ CHECK: mov	r2, r4                  @ encoding: [0x04,0x20,0xa0,0xe1]
@ CHECK: lsl	r4, r4, #1              @ encoding: [0x84,0x40,0xa0,0xe1]


@------------------------------------------------------------------------------
@ LSR
@------------------------------------------------------------------------------
	lsr r2, r4, #32
	lsr r2, r4, #2
	lsr r2, r4, #0
	lsr r4, #2

@ CHECK: lsr	r2, r4, #32             @ encoding: [0x24,0x20,0xa0,0xe1]
@ CHECK: lsr	r2, r4, #2              @ encoding: [0x24,0x21,0xa0,0xe1]
@ CHECK: mov	r2, r4                  @ encoding: [0x04,0x20,0xa0,0xe1]
@ CHECK: lsr	r4, r4, #2              @ encoding: [0x24,0x41,0xa0,0xe1]


@------------------------------------------------------------------------------
@ MCR/MCR2
@------------------------------------------------------------------------------
        mcr  p7, #1, r5, c1, c1, #4
        mcr2  p7, #1, r5, c1, c1, #4
        MCR  P7, #1, R5, C1, C1, #4
        MCR2  P7, #1, R5, C1, C1, #4

@ CHECK: mcr  p7, #1, r5, c1, c1, #4    @ encoding: [0x91,0x57,0x21,0xee]
@ CHECK: mcr2  p7, #1, r5, c1, c1, #4   @ encoding: [0x91,0x57,0x21,0xfe]
@ CHECK: mcr  p7, #1, r5, c1, c1, #4    @ encoding: [0x91,0x57,0x21,0xee]
@ CHECK: mcr2  p7, #1, r5, c1, c1, #4   @ encoding: [0x91,0x57,0x21,0xfe]

        mcrls  p7, #1, r5, c1, c1, #4
        MCRLS  P7, #1, R5, C1, C1, #4
@ CHECK: mcrls  p7, #1, r5, c1, c1, #4   @ encoding: [0x91,0x57,0x21,0x9e]
@ CHECK: mcrls  p7, #1, r5, c1, c1, #4   @ encoding: [0x91,0x57,0x21,0x9e]

@------------------------------------------------------------------------------
@ MCRR/MCRR2
@------------------------------------------------------------------------------
        mcrr  p7, #15, r5, r4, c1
        mcrr2  p7, #15, r5, r4, c1
        MCRR  P7, #15, R5, R4, C1
        MCRR2  P7, #15, R5, R4, C1

@ CHECK: mcrr  p7, #15, r5, r4, c1      @ encoding: [0xf1,0x57,0x44,0xec]
@ CHECK: mcrr2  p7, #15, r5, r4, c1     @ encoding: [0xf1,0x57,0x44,0xfc]
@ CHECK: mcrr  p7, #15, r5, r4, c1      @ encoding: [0xf1,0x57,0x44,0xec]
@ CHECK: mcrr2  p7, #15, r5, r4, c1     @ encoding: [0xf1,0x57,0x44,0xfc]

        mcrrgt  p7, #15, r5, r4, c1
        MCRRGT  P7, #15, R5, R4, C1
@ CHECK: mcrrgt  p7, #15, r5, r4, c1     @ encoding: [0xf1,0x57,0x44,0xcc]
@ CHECK: mcrrgt  p7, #15, r5, r4, c1     @ encoding: [0xf1,0x57,0x44,0xcc]

@------------------------------------------------------------------------------
@ MLA
@------------------------------------------------------------------------------
        mla  r1,r2,r3,r4
        mlas r1,r2,r3,r4
        mlane  r1,r2,r3,r4
        mlasne r1,r2,r3,r4

@ CHECK: mla 	r1, r2, r3, r4          @ encoding: [0x92,0x43,0x21,0xe0]
@ CHECK: mlas	r1, r2, r3, r4          @ encoding: [0x92,0x43,0x31,0xe0]
@ CHECK: mlane 	r1, r2, r3, r4          @ encoding: [0x92,0x43,0x21,0x10]
@ CHECK: mlasne	r1, r2, r3, r4          @ encoding: [0x92,0x43,0x31,0x10]

@------------------------------------------------------------------------------
@ MLS
@------------------------------------------------------------------------------
        mls  r2,r5,r6,r3
        mlsne  r2,r5,r6,r3

@ CHECK: mls	r2, r5, r6, r3          @ encoding: [0x95,0x36,0x62,0xe0]
@ CHECK: mlsne	r2, r5, r6, r3          @ encoding: [0x95,0x36,0x62,0x10]

@------------------------------------------------------------------------------
@ MOV (immediate)
@------------------------------------------------------------------------------
        mov r3, #7
        mov r3, $7
        mov r3, 7
        mov r3, -7
        mov r4, #0xff0
        mov r5, #0xff0000
        mov r7, #42, #0
        mov r7, #42, #10
	mov r7, #(0xff << 16)
        mov r7, #-2147483638
        mov r7, #42, #2
        mov pc, #42, #2
        mov r7, #0, #2
        mov r7, #40, #2
        mov r7, $40, $2
        mov r7, 40, 2
        mov r7, (2 * 20), (1 << 1)
        mov r7, #42, #30
        mov r6, #0xffff
        movw r9, #0xffff
        movs r3, #7
        moveq r4, #0xff0
        movseq r5, #0xff0000

@ CHECK: mov	r3, #7                  @ encoding: [0x07,0x30,0xa0,0xe3]
@ CHECK: mov	r3, #7                  @ encoding: [0x07,0x30,0xa0,0xe3]
@ CHECK: mov	r3, #7                  @ encoding: [0x07,0x30,0xa0,0xe3]
@ CHECK: mvn	r3, #6                  @ encoding: [0x06,0x30,0xe0,0xe3]
@ CHECK: mov	r4, #4080               @ encoding: [0xff,0x4e,0xa0,0xe3]
@ CHECK: mov	r5, #16711680           @ encoding: [0xff,0x58,0xa0,0xe3]
@ CHECK: mov    r7, #42                 @ encoding: [0x2a,0x70,0xa0,0xe3]
@ CHECK: mov    r7, #176160768          @ encoding: [0x2a,0x75,0xa0,0xe3]
@ CHECK: mov	r7, #16711680           @ encoding: [0xff,0x78,0xa0,0xe3]
@ CHECK: mov    r7, #-2147483638        @ encoding: [0x2a,0x71,0xa0,0xe3]
@ CHECK: mov    r7, #-2147483638        @ encoding: [0x2a,0x71,0xa0,0xe3]
@ CHECK: mov    pc, #2147483658         @ encoding: [0x2a,0xf1,0xa0,0xe3]
@ CHECK: mov    r7, #0, #2              @ encoding: [0x00,0x71,0xa0,0xe3]
@ CHECK: mov    r7, #40, #2             @ encoding: [0x28,0x71,0xa0,0xe3]
@ CHECK: mov    r7, #40, #2             @ encoding: [0x28,0x71,0xa0,0xe3]
@ CHECK: mov    r7, #40, #2             @ encoding: [0x28,0x71,0xa0,0xe3]
@ CHECK: mov    r7, #40, #2             @ encoding: [0x28,0x71,0xa0,0xe3]
@ CHECK: mov    r7, #42, #30            @ encoding: [0x2a,0x7f,0xa0,0xe3]
@ CHECK: movw	r6, #65535              @ encoding: [0xff,0x6f,0x0f,0xe3]
@ CHECK: movw	r9, #65535              @ encoding: [0xff,0x9f,0x0f,0xe3]
@ CHECK: movs	r3, #7                  @ encoding: [0x07,0x30,0xb0,0xe3]
@ CHECK: moveq	r4, #4080               @ encoding: [0xff,0x4e,0xa0,0x03]
@ CHECK: movseq	r5, #16711680           @ encoding: [0xff,0x58,0xb0,0x03]

@------------------------------------------------------------------------------
@ MOV (register)
@------------------------------------------------------------------------------
        mov r2, r3
        movs r2, r3
        moveq r2, r3
        movseq r2, r3
        mov r12, r8, lsl #(2 - 2)
        lsl r2, r3, #(2 - 2)
        mov r12, r8, lsr #(2 - 2)
        lsr r2, r3, #(2 - 2)
        mov r12, r8, asr #(2 - 2)
        asr r2, r3, #(2 - 2)
        mov r12, r8, ror #(2 - 2)
        ror r2, r3, #(2 - 2)

@ CHECK: mov	r2, r3                  @ encoding: [0x03,0x20,0xa0,0xe1]
@ CHECK: movs	r2, r3                  @ encoding: [0x03,0x20,0xb0,0xe1]
@ CHECK: moveq	r2, r3                  @ encoding: [0x03,0x20,0xa0,0x01]
@ CHECK: movseq	r2, r3                  @ encoding: [0x03,0x20,0xb0,0x01]
@ CHECK: mov	r12, r8                 @ encoding: [0x08,0xc0,0xa0,0xe1]
@ CHECK: mov	r2, r3                  @ encoding: [0x03,0x20,0xa0,0xe1]
@ CHECK: mov	r12, r8                 @ encoding: [0x08,0xc0,0xa0,0xe1]
@ CHECK: mov	r2, r3                  @ encoding: [0x03,0x20,0xa0,0xe1]
@ CHECK: mov	r12, r8                 @ encoding: [0x08,0xc0,0xa0,0xe1]
@ CHECK: mov	r2, r3                  @ encoding: [0x03,0x20,0xa0,0xe1]
@ CHECK: mov	r12, r8                 @ encoding: [0x08,0xc0,0xa0,0xe1]
@ CHECK: mov	r2, r3                  @ encoding: [0x03,0x20,0xa0,0xe1]


@------------------------------------------------------------------------------
@ MOVT
@------------------------------------------------------------------------------
        movt r3, #7
        movt r6, #0xffff
        movteq r4, #0xff0

@ CHECK: movt	r3, #7                  @ encoding: [0x07,0x30,0x40,0xe3]
@ CHECK: movt	r6, #65535              @ encoding: [0xff,0x6f,0x4f,0xe3]
@ CHECK: movteq	r4, #4080               @ encoding: [0xf0,0x4f,0x40,0x03]


@------------------------------------------------------------------------------
@ MRC/MRC2
@------------------------------------------------------------------------------
        mrc  p14, #0, r1, c1, c2, #4
        mrc  p15, #7, apsr_nzcv, c15, c6, #6
        mrc2  p14, #0, r1, c1, c2, #4
        mrc2  p9, #7, apsr_nzcv, c15, c0, #1
        MRC  P14, #0, R1, C1, C2, #4
        MRC  P15, #7, APSR_NZCV, C15, C6, #6
        MRC2  P14, #0, R1, C1, C2, #4
        MRC2  P9, #7, APSR_NZCV, C15, C0, #1

@ CHECK: mrc  p14, #0, r1, c1, c2, #4             @ encoding: [0x92,0x1e,0x11,0xee]
@ CHECK: mrc  p15, #7, apsr_nzcv, c15, c6, #6     @ encoding: [0xd6,0xff,0xff,0xee]
@ CHECK: mrc2  p14, #0, r1, c1, c2, #4            @ encoding: [0x92,0x1e,0x11,0xfe]
@ CHECK: mrc2  p9, #7, apsr_nzcv, c15, c0, #1     @ encoding: [0x30,0xf9,0xff,0xfe]
@ CHECK: mrc  p14, #0, r1, c1, c2, #4             @ encoding: [0x92,0x1e,0x11,0xee]
@ CHECK: mrc  p15, #7, apsr_nzcv, c15, c6, #6     @ encoding: [0xd6,0xff,0xff,0xee]
@ CHECK: mrc2  p14, #0, r1, c1, c2, #4            @ encoding: [0x92,0x1e,0x11,0xfe]
@ CHECK: mrc2  p9, #7, apsr_nzcv, c15, c0, #1     @ encoding: [0x30,0xf9,0xff,0xfe]

         mrceq  p15, #7, apsr_nzcv, c15, c6, #6
         MRCEQ  P15, #7, APSR_NZCV, C15, C6, #6
@ CHECK: mrceq  p15, #7, apsr_nzcv, c15, c6, #6   @ encoding: [0xd6,0xff,0xff,0x0e]
@ CHECK: mrceq  p15, #7, apsr_nzcv, c15, c6, #6   @ encoding: [0xd6,0xff,0xff,0x0e]

@------------------------------------------------------------------------------
@ MRRC/MRRC2
@------------------------------------------------------------------------------
        mrrc  p7, #1, r5, r4, c1
        mrrc2  p7, #1, r5, r4, c1
        MRRC  P7, #1, R5, R4, C1
        MRRC2  P7, #1, R5, R4, C1

@ CHECK: mrrc  p7, #1, r5, r4, c1       @ encoding: [0x11,0x57,0x54,0xec]
@ CHECK: mrrc2  p7, #1, r5, r4, c1      @ encoding: [0x11,0x57,0x54,0xfc]
@ CHECK: mrrc  p7, #1, r5, r4, c1       @ encoding: [0x11,0x57,0x54,0xec]
@ CHECK: mrrc2  p7, #1, r5, r4, c1      @ encoding: [0x11,0x57,0x54,0xfc]

        mrrclo  p7, #1, r5, r4, c1
        MRRCLO  P7, #1, R5, R4, C1
@ CHECK: mrrclo  p7, #1, r5, r4, c1      @ encoding: [0x11,0x57,0x54,0x3c]
@ CHECK: mrrclo  p7, #1, r5, r4, c1      @ encoding: [0x11,0x57,0x54,0x3c]

@------------------------------------------------------------------------------
@ MRS
@------------------------------------------------------------------------------
        mrs  r8, apsr
        mrs  r8, cpsr
        mrs  r8, spsr
@ CHECK: mrs  r8, apsr                  @ encoding: [0x00,0x80,0x0f,0xe1]
@ CHECK: mrs  r8, apsr                  @ encoding: [0x00,0x80,0x0f,0xe1]
@ CHECK: mrs  r8, spsr                  @ encoding: [0x00,0x80,0x4f,0xe1]



@------------------------------------------------------------------------------
@ MSR
@------------------------------------------------------------------------------

        msr  apsr, #5
        msr  apsr, $5
        msr  apsr, 5
        msr  apsr_g, #5
        msr  apsr_nzcvq, #5
        msr  APSR_nzcvq, #5
        msr  apsr_nzcvqg, #5
        msr  cpsr_fc, #5
        msr  cpsr_c, #5
        msr  cpsr_x, #5
        msr  cpsr_fc, #5
        msr  cpsr_all, #5
        msr  cpsr_fsx, #5
        msr  spsr_fc, #5
        msr  SPSR_fsxc, #5
        msr  cpsr_fsxc, #5
	msr  apsr_nzcvqg, #(0xff << 16)
        msr  APSR_nzcvq, #42, #2
        msr  apsr_nzcvqg, #2147483658
        msr  SPSR_fsxc, #40, #2
        msr  SPSR_fsxc, $40, $2
        msr  SPSR_fsxc, 40, 2
        msr  SPSR_fsxc, (2 * 20), (1 << 1)

@ CHECK: msr	APSR_nzcvq, #5          @ encoding: [0x05,0xf0,0x28,0xe3]
@ CHECK: msr	APSR_nzcvq, #5          @ encoding: [0x05,0xf0,0x28,0xe3]
@ CHECK: msr	APSR_nzcvq, #5          @ encoding: [0x05,0xf0,0x28,0xe3]
@ CHECK: msr	APSR_g, #5              @ encoding: [0x05,0xf0,0x24,0xe3]
@ CHECK: msr	APSR_nzcvq, #5          @ encoding: [0x05,0xf0,0x28,0xe3]
@ CHECK: msr	APSR_nzcvq, #5          @ encoding: [0x05,0xf0,0x28,0xe3]
@ CHECK: msr	APSR_nzcvqg, #5         @ encoding: [0x05,0xf0,0x2c,0xe3]
@ CHECK: msr	CPSR_fc, #5             @ encoding: [0x05,0xf0,0x29,0xe3]
@ CHECK: msr	CPSR_c, #5              @ encoding: [0x05,0xf0,0x21,0xe3]
@ CHECK: msr	CPSR_x, #5              @ encoding: [0x05,0xf0,0x22,0xe3]
@ CHECK: msr	CPSR_fc, #5             @ encoding: [0x05,0xf0,0x29,0xe3]
@ CHECK: msr	CPSR_fc, #5             @ encoding: [0x05,0xf0,0x29,0xe3]
@ CHECK: msr	CPSR_fsx, #5            @ encoding: [0x05,0xf0,0x2e,0xe3]
@ CHECK: msr	SPSR_fc, #5             @ encoding: [0x05,0xf0,0x69,0xe3]
@ CHECK: msr	SPSR_fsxc, #5           @ encoding: [0x05,0xf0,0x6f,0xe3]
@ CHECK: msr	CPSR_fsxc, #5           @ encoding: [0x05,0xf0,0x2f,0xe3]
@ CHECK: msr	APSR_nzcvqg, #16711680  @ encoding: [0xff,0xf8,0x2c,0xe3]
@ CHECK: msr    APSR_nzcvq, #2147483658 @ encoding: [0x2a,0xf1,0x28,0xe3]
@ CHECK: msr    APSR_nzcvqg, #2147483658 @ encoding: [0x2a,0xf1,0x2c,0xe3]
@ CHECK: msr    SPSR_fsxc, #40, #2      @ encoding: [0x28,0xf1,0x6f,0xe3]
@ CHECK: msr    SPSR_fsxc, #40, #2      @ encoding: [0x28,0xf1,0x6f,0xe3]
@ CHECK: msr    SPSR_fsxc, #40, #2      @ encoding: [0x28,0xf1,0x6f,0xe3]
@ CHECK: msr    SPSR_fsxc, #40, #2      @ encoding: [0x28,0xf1,0x6f,0xe3]

        msr  apsr, r0
        msr  apsr_g, r0
        msr  apsr_nzcvq, r0
        msr  APSR_nzcvq, r0
        msr  apsr_nzcvqg, r0
        msr  cpsr_fc, r0
        msr  cpsr_c, r0
        msr  cpsr_x, r0
        msr  cpsr_fc, r0
        msr  cpsr_all, r0
        msr  cpsr_fsx, r0
        msr  spsr_fc, r0
        msr  SPSR_fsxc, r0
        msr  cpsr_fsxc, r0

@ CHECK: msr  APSR_nzcvq, r0            @ encoding: [0x00,0xf0,0x28,0xe1]
@ CHECK: msr  APSR_g, r0                @ encoding: [0x00,0xf0,0x24,0xe1]
@ CHECK: msr  APSR_nzcvq, r0            @ encoding: [0x00,0xf0,0x28,0xe1]
@ CHECK: msr  APSR_nzcvq, r0            @ encoding: [0x00,0xf0,0x28,0xe1]
@ CHECK: msr  APSR_nzcvqg, r0           @ encoding: [0x00,0xf0,0x2c,0xe1]
@ CHECK: msr  CPSR_fc, r0               @ encoding: [0x00,0xf0,0x29,0xe1]
@ CHECK: msr  CPSR_c, r0                @ encoding: [0x00,0xf0,0x21,0xe1]
@ CHECK: msr  CPSR_x, r0                @ encoding: [0x00,0xf0,0x22,0xe1]
@ CHECK: msr  CPSR_fc, r0               @ encoding: [0x00,0xf0,0x29,0xe1]
@ CHECK: msr  CPSR_fc, r0               @ encoding: [0x00,0xf0,0x29,0xe1]
@ CHECK: msr  CPSR_fsx, r0              @ encoding: [0x00,0xf0,0x2e,0xe1]
@ CHECK: msr  SPSR_fc, r0               @ encoding: [0x00,0xf0,0x69,0xe1]
@ CHECK: msr  SPSR_fsxc, r0             @ encoding: [0x00,0xf0,0x6f,0xe1]
@ CHECK: msr  CPSR_fsxc, r0             @ encoding: [0x00,0xf0,0x2f,0xe1]

@------------------------------------------------------------------------------
@ MUL
@------------------------------------------------------------------------------
        mul r5, r6, r7
        muls r5, r6, r7
        mulgt r5, r6, r7
        mulsle r5, r6, r7
        mul r11, r5

@ CHECK: mul	r5, r6, r7              @ encoding: [0x96,0x07,0x05,0xe0]
@ CHECK: muls	r5, r6, r7              @ encoding: [0x96,0x07,0x15,0xe0]
@ CHECK: mulgt	r5, r6, r7              @ encoding: [0x96,0x07,0x05,0xc0]
@ CHECK: mulsle	r5, r6, r7              @ encoding: [0x96,0x07,0x15,0xd0]


@------------------------------------------------------------------------------
@ MVN (immediate)
@------------------------------------------------------------------------------
        mvn r3, #7
        mvn r3, $7
        mvn r3, 7
        mvn r3, -7
        mvn r7, #~0xffffff00
        mvn r4, #0xff0
        mvn r5, #0xff0000
	mvn r7, #(0xff << 16)
        mvn r7, #-2147483638
        mvn r7, #42, #2
        mvn r7, #40, #2
        mvn r7, $40, $2
        mvn r7, 40, 2
        mvn r7, (2 * 20), (1 << 1)
        mvns r3, #7
        mvneq r4, #0xff0
        mvnseq r5, #0xff0000

@ CHECK: mvn	r3, #7                  @ encoding: [0x07,0x30,0xe0,0xe3]
@ CHECK: mvn	r3, #7                  @ encoding: [0x07,0x30,0xe0,0xe3]
@ CHECK: mvn	r3, #7                  @ encoding: [0x07,0x30,0xe0,0xe3]
@ CHECK: mov	r3, #6                  @ encoding: [0x06,0x30,0xa0,0xe3]
@ CHECK: mvn    r7, #255                @ encoding: [0xff,0x70,0xe0,0xe3]
@ CHECK: mvn	r4, #4080               @ encoding: [0xff,0x4e,0xe0,0xe3]
@ CHECK: mvn	r5, #16711680           @ encoding: [0xff,0x58,0xe0,0xe3]
@ CHECK: mvn	r7, #16711680           @ encoding: [0xff,0x78,0xe0,0xe3]
@ CHECK: mvn    r7, #-2147483638        @ encoding: [0x2a,0x71,0xe0,0xe3]
@ CHECK: mvn    r7, #-2147483638        @ encoding: [0x2a,0x71,0xe0,0xe3]
@ CHECK: mvn    r7, #40, #2             @ encoding: [0x28,0x71,0xe0,0xe3]
@ CHECK: mvn    r7, #40, #2             @ encoding: [0x28,0x71,0xe0,0xe3]
@ CHECK: mvn    r7, #40, #2             @ encoding: [0x28,0x71,0xe0,0xe3]
@ CHECK: mvn    r7, #40, #2             @ encoding: [0x28,0x71,0xe0,0xe3]
@ CHECK: mvns	r3, #7                  @ encoding: [0x07,0x30,0xf0,0xe3]
@ CHECK: mvneq	r4, #4080               @ encoding: [0xff,0x4e,0xe0,0x03]
@ CHECK: mvnseq	r5, #16711680           @ encoding: [0xff,0x58,0xf0,0x03]


@------------------------------------------------------------------------------
@ MVN (register)
@------------------------------------------------------------------------------
        mvn r2, r3
        mvns r2, r3
        mvn r5, r6, lsl #19
        mvn r5, r6, lsr #9
        mvn r5, r6, asr #4
        mvn r5, r6, ror #6
        mvn r5, r6, rrx
        mvneq r2, r3
        mvnseq r2, r3, lsl #10

@ CHECK: mvn	r2, r3                  @ encoding: [0x03,0x20,0xe0,0xe1]
@ CHECK: mvns	r2, r3                  @ encoding: [0x03,0x20,0xf0,0xe1]
@ CHECK: mvn	r5, r6, lsl #19         @ encoding: [0x86,0x59,0xe0,0xe1]
@ CHECK: mvn	r5, r6, lsr #9          @ encoding: [0xa6,0x54,0xe0,0xe1]
@ CHECK: mvn	r5, r6, asr #4          @ encoding: [0x46,0x52,0xe0,0xe1]
@ CHECK: mvn	r5, r6, ror #6          @ encoding: [0x66,0x53,0xe0,0xe1]
@ CHECK: mvn	r5, r6, rrx             @ encoding: [0x66,0x50,0xe0,0xe1]
@ CHECK: mvneq	r2, r3                  @ encoding: [0x03,0x20,0xe0,0x01]
@ CHECK: mvnseq	r2, r3, lsl #10         @ encoding: [0x03,0x25,0xf0,0x01]


@------------------------------------------------------------------------------
@ MVN (shifted register)
@------------------------------------------------------------------------------
        mvn r5, r6, lsl r7
        mvns r5, r6, lsr r7
        mvngt r5, r6, asr r7
        mvnslt r5, r6, ror r7

@ CHECK: mvn	r5, r6, lsl r7          @ encoding: [0x16,0x57,0xe0,0xe1]
@ CHECK: mvns	r5, r6, lsr r7          @ encoding: [0x36,0x57,0xf0,0xe1]
@ CHECK: mvngt	r5, r6, asr r7          @ encoding: [0x56,0x57,0xe0,0xc1]
@ CHECK: mvnslt	r5, r6, ror r7          @ encoding: [0x76,0x57,0xf0,0xb1]

@------------------------------------------------------------------------------
@ NEG
@------------------------------------------------------------------------------
        neg r5, r8

@ CHECK: rsb	r5, r8, #0              @ encoding: [0x00,0x50,0x68,0xe2]


@------------------------------------------------------------------------------
@ NOP
@------------------------------------------------------------------------------
        nop
        nop.w
        nopgt

@ CHECK: nop @ encoding: [0x00,0xf0,0x20,0xe3]
@ CHECK: nop @ encoding: [0x00,0xf0,0x20,0xe3]
@ CHECK: nopgt @ encoding: [0x00,0xf0,0x20,0xc3]


@------------------------------------------------------------------------------
@ ORR
@------------------------------------------------------------------------------
        orr r4, r5, #0xf000
        orr r4, r5, $0xf000
        orr r4, r5, 0xf000
	orr r7, r8, #(0xff << 16)
        orr r7, r8, #-2147483638
        orr r7, r8, #42, #2
        orr r7, r8, #40, #2
        orr r7, r8, $40, $2
        orr r7, r8, 40, 2
        orr r7, r8, (2 * 20), (1 << 1)
        orr r4, r5, r6
        orr r4, r5, r6, lsl #5
        orr r4, r5, r6, lsr #5
        orr r4, r5, r6, lsr #5
        orr r4, r5, r6, asr #5
        orr r4, r5, r6, ror #5
        orr r6, r7, r8, lsl r9
        orr r6, r7, r8, lsr r9
        orr r6, r7, r8, asr r9
        orr r6, r7, r8, ror r9
        orr r4, r5, r6, rrx

        @ destination register is optional
        orr r5, #0xf000
        orr r5, $0xf000
        orr r5, 0xf000

        orr r7, #(0xff << 16)
        orr r7, #-2147483638
        orr r7, #42, #2
        orr r7, #40, #2
        orr r7, $40, $2
        orr r7, 40, 2
        orr r7, (2 * 20), (1 << 1)

        orr r4, r5
        orr r4, r5, lsl #5
        orr r4, r5, lsr #5
        orr r4, r5, lsr #5
        orr r4, r5, asr #5
        orr r4, r5, ror #5
        orr r6, r7, lsl r9
        orr r6, r7, lsr r9
        orr r6, r7, asr r9
        orr r6, r7, ror r9
        orr r4, r5, rrx

@ CHECK: orr	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x85,0xe3]
@ CHECK: orr	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x85,0xe3]
@ CHECK: orr	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x85,0xe3]
@ CHECK: orr	r7, r8, #16711680       @ encoding: [0xff,0x78,0x88,0xe3]
@ CHECK: orr    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x88,0xe3]
@ CHECK: orr    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x88,0xe3]
@ CHECK: orr    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x88,0xe3]
@ CHECK: orr    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x88,0xe3]
@ CHECK: orr    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x88,0xe3]
@ CHECK: orr    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x88,0xe3]
@ CHECK: orr	r4, r5, r6              @ encoding: [0x06,0x40,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x85,0xe1]
@ CHECK: orr	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x85,0xe1]
@ CHECK: orr	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x87,0xe1]
@ CHECK: orr	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x87,0xe1]
@ CHECK: orr	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x87,0xe1]
@ CHECK: orr	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x87,0xe1]
@ CHECK: orr	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x85,0xe1]

@ CHECK: orr	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0xe3]
@ CHECK: orr	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0xe3]
@ CHECK: orr	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0xe3]
@ CHECK: orr	r7, r7, #16711680       @ encoding: [0xff,0x78,0x87,0xe3]
@ CHECK: orr	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x87,0xe3]
@ CHECK: orr	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x87,0xe3]
@ CHECK: orr	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x87,0xe3]
@ CHECK: orr 	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x87,0xe3]
@ CHECK: orr 	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x87,0xe3]
@ CHECK: orr 	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x87,0xe3]
@ CHECK: orr	r4, r4, r5              @ encoding: [0x05,0x40,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x84,0xe1]
@ CHECK: orr	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x84,0xe1]
@ CHECK: orr	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x86,0xe1]
@ CHECK: orr	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x86,0xe1]
@ CHECK: orr	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x86,0xe1]
@ CHECK: orr	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x86,0xe1]
@ CHECK: orr	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x84,0xe1]

        orrseq r4, r5, #0xf000
        orrne r4, r5, r6
        orrseq r4, r5, r6, lsl #5
        orrlo r6, r7, r8, ror r9
        orrshi r4, r5, r6, rrx
        orrcs r5, #0xf000
        orrseq r4, r5
        orrne r6, r7, asr r9
        orrslt r6, r7, ror r9
        orrsgt r4, r5, rrx

@ CHECK: orrseq	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x95,0x03]
@ CHECK: orrne	r4, r5, r6              @ encoding: [0x06,0x40,0x85,0x11]
@ CHECK: orrseq	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x95,0x01]
@ CHECK: orrlo	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x87,0x31]
@ CHECK: orrshi	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x95,0x81]
@ CHECK: orrhs	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x85,0x23]
@ CHECK: orrseq	r4, r4, r5              @ encoding: [0x05,0x40,0x94,0x01]
@ CHECK: orrne	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x86,0x11]
@ CHECK: orrslt	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x96,0xb1]
@ CHECK: orrsgt	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x94,0xc1]

    @ Test right shift by 32, which is encoded as 0
    orr r3, r1, r2, lsr #32
    orr r3, r1, r2, asr #32
@ CHECK: orr	r3, r1, r2, lsr #32     @ encoding: [0x22,0x30,0x81,0xe1]
@ CHECK: orr	r3, r1, r2, asr #32     @ encoding: [0x42,0x30,0x81,0xe1]

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

@ CHECK: pkhbt	r2, r2, r3              @ encoding: [0x13,0x20,0x82,0xe6]
@ CHECK: pkhbt	r2, r2, r3, lsl #31     @ encoding: [0x93,0x2f,0x82,0xe6]
@ CHECK: pkhbt	r2, r2, r3              @ encoding: [0x13,0x20,0x82,0xe6]
@ CHECK: pkhbt	r2, r2, r3, lsl #15     @ encoding: [0x93,0x27,0x82,0xe6]

@ CHECK: pkhbt	r2, r3, r2              @ encoding: [0x12,0x20,0x83,0xe6]
@ CHECK: pkhtb	r2, r2, r3, asr #31     @ encoding: [0xd3,0x2f,0x82,0xe6]
@ CHECK: pkhtb	r2, r2, r3, asr #15     @ encoding: [0xd3,0x27,0x82,0xe6]

@------------------------------------------------------------------------------
@ FIXME: PLD
@------------------------------------------------------------------------------
@------------------------------------------------------------------------------
@ FIXME: PLI
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ POP
@------------------------------------------------------------------------------
        pop {r7}
        pop {r7, r8, r9, r10}

@ CHECK: pop	{r7}                    @ encoding: [0x04,0x70,0x9d,0xe4]
@ CHECK: pop	{r7, r8, r9, r10}       @ encoding: [0x80,0x07,0xbd,0xe8]


@------------------------------------------------------------------------------
@ PUSH
@------------------------------------------------------------------------------
        push {r7}
        push {r7, r8, r9, r10}

@ CHECK: push	{r7}                    @ encoding: [0x04,0x70,0x2d,0xe5]
@ CHECK: push	{r7, r8, r9, r10}       @ encoding: [0x80,0x07,0x2d,0xe9]


@------------------------------------------------------------------------------
@ QADD/QADD16/QADD8
@------------------------------------------------------------------------------
        qadd r1, r2, r3
        qaddne r1, r2, r3
        qadd16 r1, r2, r3
        qadd16gt r1, r2, r3
        qadd8 r1, r2, r3
        qadd8le r1, r2, r3

@ CHECK: qadd	r1, r2, r3              @ encoding: [0x52,0x10,0x03,0xe1]
@ CHECK: qaddne	r1, r2, r3              @ encoding: [0x52,0x10,0x03,0x11]
@ CHECK: qadd16	r1, r2, r3              @ encoding: [0x13,0x1f,0x22,0xe6]
@ CHECK: qadd16gt	r1, r2, r3      @ encoding: [0x13,0x1f,0x22,0xc6]
@ CHECK: qadd8	r1, r2, r3              @ encoding: [0x93,0x1f,0x22,0xe6]
@ CHECK: qadd8le r1, r2, r3             @ encoding: [0x93,0x1f,0x22,0xd6]


@------------------------------------------------------------------------------
@ QDADD/QDSUB
@------------------------------------------------------------------------------
        qdadd r6, r7, r8
        qdaddhi r6, r7, r8
        qdsub r6, r7, r8
        qdsubhi r6, r7, r8

@ CHECK: qdadd	r6, r7, r8              @ encoding: [0x57,0x60,0x48,0xe1]
@ CHECK: qdaddhi r6, r7, r8             @ encoding: [0x57,0x60,0x48,0x81]
@ CHECK: qdsub	r6, r7, r8              @ encoding: [0x57,0x60,0x68,0xe1]
@ CHECK: qdsubhi r6, r7, r8             @ encoding: [0x57,0x60,0x68,0x81]


@------------------------------------------------------------------------------
@ QSAX
@------------------------------------------------------------------------------
        qsax r9, r12, r0
        qsaxeq r9, r12, r0

@ CHECK: qsax	r9, r12, r0             @ encoding: [0x50,0x9f,0x2c,0xe6]
@ CHECK: qsaxeq	r9, r12, r0             @ encoding: [0x50,0x9f,0x2c,0x06]


@------------------------------------------------------------------------------
@ QSUB/QSUB16/QSUB8
@------------------------------------------------------------------------------
        qsub r1, r2, r3
        qsubne r1, r2, r3
        qsub16 r1, r2, r3
        qsub16gt r1, r2, r3
        qsub8 r1, r2, r3
        qsub8le r1, r2, r3

@ CHECK: qsub	r1, r2, r3              @ encoding: [0x52,0x10,0x23,0xe1]
@ CHECK: qsubne	r1, r2, r3              @ encoding: [0x52,0x10,0x23,0x11]
@ CHECK: qsub16	r1, r2, r3              @ encoding: [0x73,0x1f,0x22,0xe6]
@ CHECK: qsub16gt	r1, r2, r3      @ encoding: [0x73,0x1f,0x22,0xc6]
@ CHECK: qsub8	r1, r2, r3              @ encoding: [0xf3,0x1f,0x22,0xe6]
@ CHECK: qsub8le r1, r2, r3             @ encoding: [0xf3,0x1f,0x22,0xd6]


@------------------------------------------------------------------------------
@ RBIT
@------------------------------------------------------------------------------
        rbit r1, r2
        rbitne r1, r2

@ CHECK: rbit	r1, r2                  @ encoding: [0x32,0x1f,0xff,0xe6]
@ CHECK: rbitne	r1, r2                  @ encoding: [0x32,0x1f,0xff,0x16]


@------------------------------------------------------------------------------
@ REV/REV16/REVSH
@------------------------------------------------------------------------------
        rev r1, r9
        revne r1, r5
        rev16 r8, r3
        rev16ne r12, r4
        revsh r4, r9
        revshne r9, r1

@ CHECK: rev	r1, r9                  @ encoding: [0x39,0x1f,0xbf,0xe6]
@ CHECK: revne	r1, r5                  @ encoding: [0x35,0x1f,0xbf,0x16]
@ CHECK: rev16	r8, r3                  @ encoding: [0xb3,0x8f,0xbf,0xe6]
@ CHECK: rev16ne r12, r4                @ encoding: [0xb4,0xcf,0xbf,0x16]
@ CHECK: revsh	r4, r9                  @ encoding: [0xb9,0x4f,0xff,0xe6]
@ CHECK: revshne r9, r1                 @ encoding: [0xb1,0x9f,0xff,0x16]


@------------------------------------------------------------------------------
@ RFE
@------------------------------------------------------------------------------
        rfeda r2
        rfedb r3
        rfeia r5
        rfeib r6

        rfeda r4!
        rfedb r7!
        rfeia r9!
        rfeib r8!

        rfefa r2
        rfeea r3
        rfefd r5
        rfeed r6

        rfefa r4!
        rfeea r7!
        rfefd r9!
        rfeed r8!

        rfe r1
        rfe r1!

@ CHECK: rfeda	r2                      @ encoding: [0x00,0x0a,0x12,0xf8]
@ CHECK: rfedb	r3                      @ encoding: [0x00,0x0a,0x13,0xf9]
@ CHECK: rfeia	r5                      @ encoding: [0x00,0x0a,0x95,0xf8]
@ CHECK: rfeib	r6                      @ encoding: [0x00,0x0a,0x96,0xf9]

@ CHECK: rfeda	r4!                     @ encoding: [0x00,0x0a,0x34,0xf8]
@ CHECK: rfedb	r7!                     @ encoding: [0x00,0x0a,0x37,0xf9]
@ CHECK: rfeia	r9!                     @ encoding: [0x00,0x0a,0xb9,0xf8]
@ CHECK: rfeib	r8!                     @ encoding: [0x00,0x0a,0xb8,0xf9]

@ CHECK: rfeda	r2                      @ encoding: [0x00,0x0a,0x12,0xf8]
@ CHECK: rfedb	r3                      @ encoding: [0x00,0x0a,0x13,0xf9]
@ CHECK: rfeia	r5                      @ encoding: [0x00,0x0a,0x95,0xf8]
@ CHECK: rfeib	r6                      @ encoding: [0x00,0x0a,0x96,0xf9]

@ CHECK: rfeda	r4!                     @ encoding: [0x00,0x0a,0x34,0xf8]
@ CHECK: rfedb	r7!                     @ encoding: [0x00,0x0a,0x37,0xf9]
@ CHECK: rfeia	r9!                     @ encoding: [0x00,0x0a,0xb9,0xf8]
@ CHECK: rfeib	r8!                     @ encoding: [0x00,0x0a,0xb8,0xf9]

@ CHECK: rfeia	r1                      @ encoding: [0x00,0x0a,0x91,0xf8]
@ CHECK: rfeia	r1!                     @ encoding: [0x00,0x0a,0xb1,0xf8]


@------------------------------------------------------------------------------
@ ROR
@------------------------------------------------------------------------------
	ror r2, r4, #31
	ror r2, r4, #1
	ror r2, r4, #0
	ror r4, #1

@ CHECK: ror	r2, r4, #31             @ encoding: [0xe4,0x2f,0xa0,0xe1]
@ CHECK: ror	r2, r4, #1              @ encoding: [0xe4,0x20,0xa0,0xe1]
@ CHECK: mov	r2, r4                  @ encoding: [0x04,0x20,0xa0,0xe1]
@ CHECK: ror	r4, r4, #1              @ encoding: [0xe4,0x40,0xa0,0xe1]


@------------------------------------------------------------------------------
@ RSB
@------------------------------------------------------------------------------
        rsb r4, r5, #0xf000
        rsb r4, r5, $0xf000
        rsb r4, r5, 0xf000
        rsb r7, r8, #(0xff << 16)
        rsb r7, r8, #-2147483638
        rsb r7, r8, #42, #2
        rsb r7, r8, #40, #2
        rsb r7, r8, $40, $2
        rsb r7, r8, 40, 2
        rsb r7, r8, (2 * 20), (1 << 1)
        rsb r4, r5, r6
        rsb r4, r5, r6, lsl #5
        rsblo r4, r5, r6, lsr #5
        rsb r4, r5, r6, lsr #5
        rsb r4, r5, r6, asr #5
        rsb r4, r5, r6, ror #5
        rsb r6, r7, r8, lsl r9
        rsb r6, r7, r8, lsr r9
        rsb r6, r7, r8, asr r9
        rsble r6, r7, r8, ror r9
        rsb r4, r5, r6, rrx

        @ destination register is optional
        rsb r5, #0xf000
        rsb r5, $0xf000
        rsb r5, 0xf000
        rsb r7, #(0xff << 16)
        rsb r7, #-2147483638
        rsb r7, #42, #2
        rsb r7, #40, #2
        rsb r7, $40, $2
        rsb r7, 40, 2
        rsb r7, (2 * 20), (1 << 1)
        rsb r4, r5
        rsb r4, r5, lsl #5
        rsb r4, r5, lsr #5
        rsbne r4, r5, lsr #5
        rsb r4, r5, asr #5
        rsb r4, r5, ror #5
        rsbgt r6, r7, lsl r9
        rsb r6, r7, lsr r9
        rsb r6, r7, asr r9
        rsb r6, r7, ror r9
        rsb r4, r5, rrx

@ CHECK: rsb	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x65,0xe2]
@ CHECK: rsb	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x65,0xe2]
@ CHECK: rsb	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x65,0xe2]
@ CHECK: rsb	r7, r8, #16711680       @ encoding: [0xff,0x78,0x68,0xe2]
@ CHECK: rsb    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x68,0xe2]
@ CHECK: rsb    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x68,0xe2]
@ CHECK: rsb    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x68,0xe2]
@ CHECK: rsb    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x68,0xe2]
@ CHECK: rsb    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x68,0xe2]
@ CHECK: rsb    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x68,0xe2]
@ CHECK: rsb	r4, r5, r6              @ encoding: [0x06,0x40,0x65,0xe0]
@ CHECK: rsb	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x65,0xe0]
@ CHECK: rsblo	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x65,0x30]
@ CHECK: rsb	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x65,0xe0]
@ CHECK: rsb	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x65,0xe0]
@ CHECK: rsb	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x65,0xe0]
@ CHECK: rsb	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x67,0xe0]
@ CHECK: rsb	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x67,0xe0]
@ CHECK: rsb	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x67,0xe0]
@ CHECK: rsble	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x67,0xd0]
@ CHECK: rsb	r4, r5, r6, rrx         @ encoding: [0x66,0x40,0x65,0xe0]

@ CHECK: rsb	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x65,0xe2]
@ CHECK: rsb	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x65,0xe2]
@ CHECK: rsb	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x65,0xe2]
@ CHECK: rsb	r7, r7, #16711680       @ encoding: [0xff,0x78,0x67,0xe2]
@ CHECK: rsb	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x67,0xe2]
@ CHECK: rsb	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x67,0xe2]
@ CHECK: rsb	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x67,0xe2]
@ CHECK: rsb	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x67,0xe2]
@ CHECK: rsb	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x67,0xe2]
@ CHECK: rsb	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x67,0xe2]
@ CHECK: rsb	r4, r4, r5              @ encoding: [0x05,0x40,0x64,0xe0]
@ CHECK: rsb	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x64,0xe0]
@ CHECK: rsb	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x64,0xe0]
@ CHECK: rsbne	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x64,0x10]
@ CHECK: rsb	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x64,0xe0]
@ CHECK: rsb	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x64,0xe0]
@ CHECK: rsbgt	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x66,0xc0]
@ CHECK: rsb	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x66,0xe0]
@ CHECK: rsb	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x66,0xe0]
@ CHECK: rsb	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x66,0xe0]
@ CHECK: rsb	r4, r4, r5, rrx         @ encoding: [0x65,0x40,0x64,0xe0]

@------------------------------------------------------------------------------
@ RSBS
@------------------------------------------------------------------------------
    rsbs r7, #16711680
    rsbs r7, $16711680
    rsbs r7, 16711680
    rsbs r7, #(0xff << 16)
    rsbs r7, r8, #-2147483638
    rsbs r7, r8, #42, #2
    rsbs r7, r8, #40, #2
    rsbs r7, r8, $40, $2
    rsbs r7, r8, 40, 2
    rsbs r7, r8, (2 * 20), (1 << 1)

@ CHECK: rsbs	r7, r7, #16711680       @ encoding: [0xff,0x78,0x77,0xe2]
@ CHECK: rsbs	r7, r7, #16711680       @ encoding: [0xff,0x78,0x77,0xe2]
@ CHECK: rsbs	r7, r7, #16711680       @ encoding: [0xff,0x78,0x77,0xe2]
@ CHECK: rsbs	r7, r7, #16711680       @ encoding: [0xff,0x78,0x77,0xe2]
@ CHECK: rsbs   r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x78,0xe2]
@ CHECK: rsbs   r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x78,0xe2]
@ CHECK: rsbs   r7, r8, #40, #2         @ encoding: [0x28,0x71,0x78,0xe2]
@ CHECK: rsbs   r7, r8, #40, #2         @ encoding: [0x28,0x71,0x78,0xe2]
@ CHECK: rsbs   r7, r8, #40, #2         @ encoding: [0x28,0x71,0x78,0xe2]
@ CHECK: rsbs   r7, r8, #40, #2         @ encoding: [0x28,0x71,0x78,0xe2]

@------------------------------------------------------------------------------
@ RSC
@------------------------------------------------------------------------------
        rsc r4, r5, #0xf000
        rsc r4, r5, $0xf000
        rsc r4, r5, 0xf000
        rsc r7, r8, #(0xff << 16)
        rsc r7, r8, #-2147483638
        rsc r7, r8, #42, #2
        rsc r7, r8, #40, #2
        rsc r7, r8, $40, $2
        rsc r7, r8, 40, 2
        rsc r7, r8, (2 * 20), (1 << 1)
        rsc r4, r5, r6
        rsc r4, r5, r6, lsl #5
        rsclo r4, r5, r6, lsr #5
        rsc r4, r5, r6, lsr #5
        rsc r4, r5, r6, asr #5
        rsc r4, r5, r6, ror #5
        rsc r6, r7, r8, lsl r9
        rsc r6, r7, r8, lsr r9
        rsc r6, r7, r8, asr r9
        rscle r6, r7, r8, ror r9
        rscs r1, r8, #4064

        @ destination register is optional
        rsc r5, #0xf000
        rsc r5, $0xf000
        rsc r5, 0xf000
        rsc r7, #(0xff << 16)
        rsc r7, #-2147483638
        rsc r7, #42, #2
        rsc r7, #40, #2
        rsc r7, $40, $2
        rsc r7, 40, 2
        rsc r7, (2 * 20), (1 << 1)
        rsc r4, r5
        rsc r4, r5, lsl #5
        rsc r4, r5, lsr #5
        rscne r4, r5, lsr #5
        rsc r4, r5, asr #5
        rsc r4, r5, ror #5
        rscgt r6, r7, lsl r9
        rsc r6, r7, lsr r9
        rsc r6, r7, asr r9
        rsc r6, r7, ror r9

@ CHECK: rsc	r4, r5, #61440          @ encoding: [0x0f,0x4a,0xe5,0xe2]
@ CHECK: rsc	r4, r5, #61440          @ encoding: [0x0f,0x4a,0xe5,0xe2]
@ CHECK: rsc	r4, r5, #61440          @ encoding: [0x0f,0x4a,0xe5,0xe2]
@ CHECK: rsc    r7, r8, #16711680       @ encoding: [0xff,0x78,0xe8,0xe2]
@ CHECK: rsc    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0xe8,0xe2]
@ CHECK: rsc    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0xe8,0xe2]
@ CHECK: rsc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xe8,0xe2]
@ CHECK: rsc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xe8,0xe2]
@ CHECK: rsc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xe8,0xe2]
@ CHECK: rsc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xe8,0xe2]
@ CHECK: rsc	r4, r5, r6              @ encoding: [0x06,0x40,0xe5,0xe0]
@ CHECK: rsc	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0xe5,0xe0]
@ CHECK: rsclo	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0xe5,0x30]
@ CHECK: rsc	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0xe5,0xe0]
@ CHECK: rsc	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0xe5,0xe0]
@ CHECK: rsc	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0xe5,0xe0]
@ CHECK: rsc	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0xe7,0xe0]
@ CHECK: rsc	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0xe7,0xe0]
@ CHECK: rsc	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0xe7,0xe0]
@ CHECK: rscle	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0xe7,0xd0]
@ CHECK: rscs	r1, r8, #4064           @ encoding: [0xfe,0x1e,0xf8,0xe2]

@ CHECK: rsc	r5, r5, #61440          @ encoding: [0x0f,0x5a,0xe5,0xe2]
@ CHECK: rsc	r5, r5, #61440          @ encoding: [0x0f,0x5a,0xe5,0xe2]
@ CHECK: rsc	r5, r5, #61440          @ encoding: [0x0f,0x5a,0xe5,0xe2]
@ CHECK: rsc	r7, r7, #16711680       @ encoding: [0xff,0x78,0xe7,0xe2]
@ CHECK: rsc	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0xe7,0xe2]
@ CHECK: rsc	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0xe7,0xe2]
@ CHECK: rsc	r7, r7, #40, #2         @ encoding: [0x28,0x71,0xe7,0xe2]
@ CHECK: rsc	r7, r7, #40, #2         @ encoding: [0x28,0x71,0xe7,0xe2]
@ CHECK: rsc	r7, r7, #40, #2         @ encoding: [0x28,0x71,0xe7,0xe2]
@ CHECK: rsc	r7, r7, #40, #2         @ encoding: [0x28,0x71,0xe7,0xe2]
@ CHECK: rsc	r4, r4, r5              @ encoding: [0x05,0x40,0xe4,0xe0]
@ CHECK: rsc	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0xe4,0xe0]
@ CHECK: rsc	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0xe4,0xe0]
@ CHECK: rscne	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0xe4,0x10]
@ CHECK: rsc	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0xe4,0xe0]
@ CHECK: rsc	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0xe4,0xe0]
@ CHECK: rscgt	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0xe6,0xc0]
@ CHECK: rsc	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0xe6,0xe0]
@ CHECK: rsc	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0xe6,0xe0]
@ CHECK: rsc	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0xe6,0xe0]

@------------------------------------------------------------------------------
@ RRX/RRXS
@------------------------------------------------------------------------------

         rrx r0, r1
	 rrx sp, pc
	 rrx pc, lr
	 rrx lr, sp

@ CHECK: rrx	r0, r1                  @ encoding: [0x61,0x00,0xa0,0xe1]
@ CHECK: rrx	sp, pc                  @ encoding: [0x6f,0xd0,0xa0,0xe1]
@ CHECK: rrx	pc, lr                  @ encoding: [0x6e,0xf0,0xa0,0xe1]
@ CHECK: rrx	lr, sp                  @ encoding: [0x6d,0xe0,0xa0,0xe1]

         rrxs r0, r1
	 rrxs sp, pc
	 rrxs pc, lr
	 rrxs lr, sp

@CHECK: rrxs	r0, r1                  @ encoding: [0x61,0x00,0xb0,0xe1]
@CHECK: rrxs	sp, pc                  @ encoding: [0x6f,0xd0,0xb0,0xe1]
@CHECK: rrxs	pc, lr                  @ encoding: [0x6e,0xf0,0xb0,0xe1]
@CHECK: rrxs	lr, sp                  @ encoding: [0x6d,0xe0,0xb0,0xe1]

@ ------------------------------------------------------------------------------
@ SADD16/SADD8
@------------------------------------------------------------------------------
        sadd16 r1, r2, r3
        sadd16gt r1, r2, r3
        sadd8 r1, r2, r3
        sadd8le r1, r2, r3

@ CHECK: sadd16	r1, r2, r3              @ encoding: [0x13,0x1f,0x12,0xe6]
@ CHECK: sadd16gt	r1, r2, r3      @ encoding: [0x13,0x1f,0x12,0xc6]
@ CHECK: sadd8	r1, r2, r3              @ encoding: [0x93,0x1f,0x12,0xe6]
@ CHECK: sadd8le r1, r2, r3             @ encoding: [0x93,0x1f,0x12,0xd6]


@------------------------------------------------------------------------------
@ SASX
@------------------------------------------------------------------------------
        sasx r9, r12, r0
        sasxeq r9, r12, r0

@ CHECK: sasx	r9, r12, r0             @ encoding: [0x30,0x9f,0x1c,0xe6]
@ CHECK: sasxeq	r9, r12, r0             @ encoding: [0x30,0x9f,0x1c,0x06]


@------------------------------------------------------------------------------
@ SBC
@------------------------------------------------------------------------------
        sbc r4, r5, #0xf000
        sbc r4, r5, $0xf000
        sbc r4, r5, 0xf000
        sbc r7, r8, #(0xff << 16)
        sbc r7, r8, #-2147483638
        sbc r7, r8, #42, #2
        sbc r7, r8, #40, #2
        sbc r7, r8, $40, $2
        sbc r7, r8, 40, 2
        sbc r7, r8, (20 * 2), (1 << 1)
        sbc r4, r5, r6
        sbc r4, r5, r6, lsl #5
        sbc r4, r5, r6, lsr #5
        sbc r4, r5, r6, lsr #5
        sbc r4, r5, r6, asr #5
        sbc r4, r5, r6, ror #5
        sbc r6, r7, r8, lsl r9
        sbc r6, r7, r8, lsr r9
        sbc r6, r7, r8, asr r9
        sbc r6, r7, r8, ror r9

        @ destination register is optional
        sbc r5, #0xf000
        sbc r5, $0xf000
        sbc r5, 0xf000
        sbc r7, #(0xff << 16)
        sbc r7, #-2147483638
        sbc r7, #42, #2
        sbc r7, #40, #2
        sbc r7, $40, $2
        sbc r7, 40, 2
        sbc r7, (20 * 2), (1 << 1)
        sbc r4, r5
        sbc r4, r5, lsl #5
        sbc r4, r5, lsr #5
        sbc r4, r5, lsr #5
        sbc r4, r5, asr #5
        sbc r4, r5, ror #5
        sbc r6, r7, lsl r9
        sbc r6, r7, lsr r9
        sbc r6, r7, asr r9
        sbc r6, r7, ror r9

@ CHECK: sbc	r4, r5, #61440          @ encoding: [0x0f,0x4a,0xc5,0xe2]
@ CHECK: sbc	r4, r5, #61440          @ encoding: [0x0f,0x4a,0xc5,0xe2]
@ CHECK: sbc	r4, r5, #61440          @ encoding: [0x0f,0x4a,0xc5,0xe2]
@ CHECK: sbc	r7, r8, #16711680       @ encoding: [0xff,0x78,0xc8,0xe2]
@ CHECK: sbc    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0xc8,0xe2]
@ CHECK: sbc    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0xc8,0xe2]
@ CHECK: sbc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xc8,0xe2]
@ CHECK: sbc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xc8,0xe2]
@ CHECK: sbc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xc8,0xe2]
@ CHECK: sbc    r7, r8, #40, #2         @ encoding: [0x28,0x71,0xc8,0xe2]
@ CHECK: sbc	r4, r5, r6              @ encoding: [0x06,0x40,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0xc5,0xe0]
@ CHECK: sbc	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0xc5,0xe0]
@ CHECK: sbc	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0xc7,0xe0]
@ CHECK: sbc	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0xc7,0xe0]
@ CHECK: sbc	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0xc7,0xe0]
@ CHECK: sbc	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0xc7,0xe0]

@ CHECK: sbc	r5, r5, #61440          @ encoding: [0x0f,0x5a,0xc5,0xe2]
@ CHECK: sbc	r5, r5, #61440          @ encoding: [0x0f,0x5a,0xc5,0xe2]
@ CHECK: sbc	r5, r5, #61440          @ encoding: [0x0f,0x5a,0xc5,0xe2]
@ CHECK: sbc	r7, r7, #16711680       @ encoding: [0xff,0x78,0xc7,0xe2]
@ CHECK: sbc	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0xc7,0xe2]
@ CHECK: sbc	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0xc7,0xe2]
@ CHECK: sbc	r7, r7, #40, #2         @ encoding: [0x28,0x71,0xc7,0xe2]
@ CHECK: sbc	r7, r7, #40, #2         @ encoding: [0x28,0x71,0xc7,0xe2]
@ CHECK: sbc	r7, r7, #40, #2         @ encoding: [0x28,0x71,0xc7,0xe2]
@ CHECK: sbc	r7, r7, #40, #2         @ encoding: [0x28,0x71,0xc7,0xe2]
@ CHECK: sbc	r4, r4, r5              @ encoding: [0x05,0x40,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0xc4,0xe0]
@ CHECK: sbc	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0xc4,0xe0]
@ CHECK: sbc	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0xc6,0xe0]
@ CHECK: sbc	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0xc6,0xe0]
@ CHECK: sbc	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0xc6,0xe0]
@ CHECK: sbc	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0xc6,0xe0]


@------------------------------------------------------------------------------
@ SBFX
@------------------------------------------------------------------------------
        sbfx r4, r5, #16, #1
        sbfxgt r4, r5, #16, #16

@ CHECK: sbfx	r4, r5, #16, #1         @ encoding: [0x55,0x48,0xa0,0xe7]
@ CHECK: sbfxgt	r4, r5, #16, #16        @ encoding: [0x55,0x48,0xaf,0xc7]


@------------------------------------------------------------------------------
@ SEL
@------------------------------------------------------------------------------
        sel r9, r2, r1
        selne r9, r2, r1

@ CHECK: sel	r9, r2, r1              @ encoding: [0xb1,0x9f,0x82,0xe6]
@ CHECK: selne	r9, r2, r1              @ encoding: [0xb1,0x9f,0x82,0x16]


@------------------------------------------------------------------------------
@ SETEND
@------------------------------------------------------------------------------
        setend be
        setend BE
        setend le
        setend LE

@ CHECK: setend	be                      @ encoding: [0x00,0x02,0x01,0xf1]
@ CHECK: setend	be                      @ encoding: [0x00,0x02,0x01,0xf1]
@ CHECK: setend	le                      @ encoding: [0x00,0x00,0x01,0xf1]
@ CHECK: setend	le                      @ encoding: [0x00,0x00,0x01,0xf1]


@------------------------------------------------------------------------------
@ SEV
@------------------------------------------------------------------------------
        sev
        seveq

@ CHECK: sev                            @ encoding: [0x04,0xf0,0x20,0xe3]
@ CHECK: seveq                          @ encoding: [0x04,0xf0,0x20,0x03]


@------------------------------------------------------------------------------
@ SHADD16/SHADD8
@------------------------------------------------------------------------------
        shadd16 r4, r8, r2
        shadd16gt r4, r8, r2
        shadd8 r4, r8, r2
        shadd8gt r4, r8, r2

@ CHECK: shadd16	r4, r8, r2      @ encoding: [0x12,0x4f,0x38,0xe6]
@ CHECK: shadd16gt	r4, r8, r2      @ encoding: [0x12,0x4f,0x38,0xc6]
@ CHECK: shadd8	r4, r8, r2              @ encoding: [0x92,0x4f,0x38,0xe6]
@ CHECK: shadd8gt	r4, r8, r2      @ encoding: [0x92,0x4f,0x38,0xc6]


@------------------------------------------------------------------------------
@ SHASX
@------------------------------------------------------------------------------
        shasx r4, r8, r2
        shasxgt r4, r8, r2

@ CHECK: shasx	r4, r8, r2              @ encoding: [0x32,0x4f,0x38,0xe6]
@ CHECK: shasxgt r4, r8, r2             @ encoding: [0x32,0x4f,0x38,0xc6]


@------------------------------------------------------------------------------
@ SHSUB16/SHSUB8
@------------------------------------------------------------------------------
        shsub16 r4, r8, r2
        shsub16gt r4, r8, r2
        shsub8 r4, r8, r2
        shsub8gt r4, r8, r2

@ CHECK: shsub16	r4, r8, r2      @ encoding: [0x72,0x4f,0x38,0xe6]
@ CHECK: shsub16gt	r4, r8, r2      @ encoding: [0x72,0x4f,0x38,0xc6]
@ CHECK: shsub8	r4, r8, r2              @ encoding: [0xf2,0x4f,0x38,0xe6]
@ CHECK: shsub8gt	r4, r8, r2      @ encoding: [0xf2,0x4f,0x38,0xc6]

@------------------------------------------------------------------------------
@ SMLABB/SMLABT/SMLATB/SMLATT
@------------------------------------------------------------------------------
        smlabb r3, r1, r9, r0
        smlabt r5, r6, r4, r1
        smlatb r4, r2, r3, r2
        smlatt r8, r3, r8, r4
        smlabbge r3, r1, r9, r0
        smlabtle r5, r6, r4, r1
        smlatbne r4, r2, r3, r2
        smlatteq r8, r3, r8, r4

@ CHECK: smlabb	r3, r1, r9, r0          @ encoding: [0x81,0x09,0x03,0xe1]
@ CHECK: smlabt	r5, r6, r4, r1          @ encoding: [0xc6,0x14,0x05,0xe1]
@ CHECK: smlatb	r4, r2, r3, r2          @ encoding: [0xa2,0x23,0x04,0xe1]
@ CHECK: smlatt	r8, r3, r8, r4          @ encoding: [0xe3,0x48,0x08,0xe1]
@ CHECK: smlabbge	r3, r1, r9, r0  @ encoding: [0x81,0x09,0x03,0xa1]
@ CHECK: smlabtle	r5, r6, r4, r1  @ encoding: [0xc6,0x14,0x05,0xd1]
@ CHECK: smlatbne	r4, r2, r3, r2  @ encoding: [0xa2,0x23,0x04,0x11]
@ CHECK: smlatteq	r8, r3, r8, r4  @ encoding: [0xe3,0x48,0x08,0x01]

@------------------------------------------------------------------------------
@ SMLAD/SMLADX
@------------------------------------------------------------------------------
        smlad r2, r3, r5, r8
        smladx r2, r3, r5, r8
        smladeq r2, r3, r5, r8
        smladxhi r2, r3, r5, r8

@ CHECK: smlad	r2, r3, r5, r8          @ encoding: [0x13,0x85,0x02,0xe7]
@ CHECK: smladx	r2, r3, r5, r8          @ encoding: [0x33,0x85,0x02,0xe7]
@ CHECK: smladeq	r2, r3, r5, r8  @ encoding: [0x13,0x85,0x02,0x07]
@ CHECK: smladxhi	r2, r3, r5, r8  @ encoding: [0x33,0x85,0x02,0x87]


@------------------------------------------------------------------------------
@ SMLAL
@------------------------------------------------------------------------------
        smlal r2, r3, r5, r8
        smlals r2, r3, r5, r8
        smlaleq r2, r3, r5, r8
        smlalshi r2, r3, r5, r8

@ CHECK: smlal	r2, r3, r5, r8          @ encoding: [0x95,0x28,0xe3,0xe0]
@ CHECK: smlals	r2, r3, r5, r8          @ encoding: [0x95,0x28,0xf3,0xe0]
@ CHECK: smlaleq	r2, r3, r5, r8  @ encoding: [0x95,0x28,0xe3,0x00]
@ CHECK: smlalshi	r2, r3, r5, r8  @ encoding: [0x95,0x28,0xf3,0x80]


@------------------------------------------------------------------------------
@ SMLALBB/SMLALBT/SMLALTB/SMLALTT
@------------------------------------------------------------------------------
        smlalbb r3, r1, r9, r0
        smlalbt r5, r6, r4, r1
        smlaltb r4, r2, r3, r2
        smlaltt r8, r3, r8, r4
        smlalbbge r3, r1, r9, r0
        smlalbtle r5, r6, r4, r1
        smlaltbne r4, r2, r3, r2
        smlaltteq r8, r3, r8, r4

@ CHECK: smlalbb	r3, r1, r9, r0  @ encoding: [0x89,0x30,0x41,0xe1]
@ CHECK: smlalbt	r5, r6, r4, r1  @ encoding: [0xc4,0x51,0x46,0xe1]
@ CHECK: smlaltb	r4, r2, r3, r2  @ encoding: [0xa3,0x42,0x42,0xe1]
@ CHECK: smlaltt	r8, r3, r8, r4  @ encoding: [0xe8,0x84,0x43,0xe1]
@ CHECK: smlalbbge	r3, r1, r9, r0  @ encoding: [0x89,0x30,0x41,0xa1]
@ CHECK: smlalbtle	r5, r6, r4, r1  @ encoding: [0xc4,0x51,0x46,0xd1]
@ CHECK: smlaltbne	r4, r2, r3, r2  @ encoding: [0xa3,0x42,0x42,0x11]
@ CHECK: smlaltteq	r8, r3, r8, r4  @ encoding: [0xe8,0x84,0x43,0x01]


@------------------------------------------------------------------------------
@ SMLALD/SMLALDX
@------------------------------------------------------------------------------
        smlald r2, r3, r5, r8
        smlaldx r2, r3, r5, r8
        smlaldeq r2, r3, r5, r8
        smlaldxhi r2, r3, r5, r8

@ CHECK: smlald	r2, r3, r5, r8          @ encoding: [0x15,0x28,0x43,0xe7]
@ CHECK: smlaldx	r2, r3, r5, r8  @ encoding: [0x35,0x28,0x43,0xe7]
@ CHECK: smlaldeq	r2, r3, r5, r8  @ encoding: [0x15,0x28,0x43,0x07]
@ CHECK: smlaldxhi	r2, r3, r5, r8  @ encoding: [0x35,0x28,0x43,0x87]


@------------------------------------------------------------------------------
@ SMLAWB/SMLAWT
@------------------------------------------------------------------------------
        smlawb r2, r3, r10, r8
        smlawt r8, r3, r5, r9
        smlawbeq r2, r7, r5, r8
        smlawthi r1, r3, r0, r8

@ CHECK: smlawb	r2, r3, r10, r8         @ encoding: [0x83,0x8a,0x22,0xe1]
@ CHECK: smlawt	r8, r3, r5, r9          @ encoding: [0xc3,0x95,0x28,0xe1]
@ CHECK: smlawbeq	r2, r7, r5, r8  @ encoding: [0x87,0x85,0x22,0x01]
@ CHECK: smlawthi	r1, r3, r0, r8  @ encoding: [0xc3,0x80,0x21,0x81]


@------------------------------------------------------------------------------
@ SMLSD/SMLSDX
@------------------------------------------------------------------------------
        smlsd r2, r3, r5, r8
        smlsdx r2, r3, r5, r8
        smlsdeq r2, r3, r5, r8
        smlsdxhi r2, r3, r5, r8

@ CHECK: smlsd	r2, r3, r5, r8          @ encoding: [0x53,0x85,0x02,0xe7]
@ CHECK: smlsdx	r2, r3, r5, r8          @ encoding: [0x73,0x85,0x02,0xe7]
@ CHECK: smlsdeq	r2, r3, r5, r8  @ encoding: [0x53,0x85,0x02,0x07]
@ CHECK: smlsdxhi	r2, r3, r5, r8  @ encoding: [0x73,0x85,0x02,0x87]


@------------------------------------------------------------------------------
@ SMLSLD/SMLSLDX
@------------------------------------------------------------------------------
        smlsld r2, r9, r5, r1
        smlsldx r4, r11, r2, r8
        smlsldeq r8, r2, r5, r6
        smlsldxhi r1, r0, r3, r8

@ CHECK: smlsld	r2, r9, r5, r1          @ encoding: [0x55,0x21,0x49,0xe7]
@ CHECK: smlsldx	r4, r11, r2, r8 @ encoding: [0x72,0x48,0x4b,0xe7]
@ CHECK: smlsldeq	r8, r2, r5, r6  @ encoding: [0x55,0x86,0x42,0x07]
@ CHECK: smlsldxhi	r1, r0, r3, r8  @ encoding: [0x73,0x18,0x40,0x87]


@------------------------------------------------------------------------------
@ SMMLA/SMMLAR
@------------------------------------------------------------------------------
        smmla r1, r2, r3, r4
        smmlar r4, r3, r2, r1
        smmlalo r1, r2, r3, r4
        smmlarcs r4, r3, r2, r1

@ CHECK: smmla	r1, r2, r3, r4          @ encoding: [0x12,0x43,0x51,0xe7]
@ CHECK: smmlar	r4, r3, r2, r1          @ encoding: [0x33,0x12,0x54,0xe7]
@ CHECK: smmlalo	r1, r2, r3, r4  @ encoding: [0x12,0x43,0x51,0x37]
@ CHECK: smmlarhs	r4, r3, r2, r1  @ encoding: [0x33,0x12,0x54,0x27]


@------------------------------------------------------------------------------
@ SMMLS/SMMLSR
@------------------------------------------------------------------------------
        smmls r1, r2, r3, r4
        smmlsr r4, r3, r2, r1
        smmlslo r1, r2, r3, r4
        smmlsrcs r4, r3, r2, r1

@ CHECK: smmls	r1, r2, r3, r4          @ encoding: [0xd2,0x43,0x51,0xe7]
@ CHECK: smmlsr	r4, r3, r2, r1          @ encoding: [0xf3,0x12,0x54,0xe7]
@ CHECK: smmlslo	r1, r2, r3, r4  @ encoding: [0xd2,0x43,0x51,0x37]
@ CHECK: smmlsrhs	r4, r3, r2, r1  @ encoding: [0xf3,0x12,0x54,0x27]


@------------------------------------------------------------------------------
@ SMMUL/SMMULR
@------------------------------------------------------------------------------
        smmul r2, r3, r4
        smmulr r3, r2, r1
        smmulcc r2, r3, r4
        smmulrhs r3, r2, r1

@ CHECK: smmul	r2, r3, r4              @ encoding: [0x13,0xf4,0x52,0xe7]
@ CHECK: smmulr	r3, r2, r1              @ encoding: [0x32,0xf1,0x53,0xe7]
@ CHECK: smmullo	r2, r3, r4      @ encoding: [0x13,0xf4,0x52,0x37]
@ CHECK: smmulrhs	r3, r2, r1      @ encoding: [0x32,0xf1,0x53,0x27]


@------------------------------------------------------------------------------
@ SMUAD/SMUADX
@------------------------------------------------------------------------------
        smuad r2, r3, r4
        smuadx r3, r2, r1
        smuadlt r2, r3, r4
        smuadxge r3, r2, r1

@ CHECK: smuad	r2, r3, r4              @ encoding: [0x13,0xf4,0x02,0xe7]
@ CHECK: smuadx	r3, r2, r1              @ encoding: [0x32,0xf1,0x03,0xe7]
@ CHECK: smuadlt	r2, r3, r4      @ encoding: [0x13,0xf4,0x02,0xb7]
@ CHECK: smuadxge	r3, r2, r1      @ encoding: [0x32,0xf1,0x03,0xa7]


@------------------------------------------------------------------------------
@ SMULBB/SMULBT/SMULTB/SMULTT
@------------------------------------------------------------------------------
        smulbb r3, r9, r0
        smulbt r5, r4, r1
        smultb r4, r2, r2
        smultt r8, r3, r4
        smulbbge r1, r9, r0
        smulbtle r5, r6, r4
        smultbne r2, r3, r2
        smultteq r8, r3, r4

@ CHECK: smulbb	r3, r9, r0              @ encoding: [0x89,0x00,0x63,0xe1]
@ CHECK: smulbt	r5, r4, r1              @ encoding: [0xc4,0x01,0x65,0xe1]
@ CHECK: smultb	r4, r2, r2              @ encoding: [0xa2,0x02,0x64,0xe1]
@ CHECK: smultt	r8, r3, r4              @ encoding: [0xe3,0x04,0x68,0xe1]
@ CHECK: smulbbge	r1, r9, r0      @ encoding: [0x89,0x00,0x61,0xa1]
@ CHECK: smulbtle	r5, r6, r4      @ encoding: [0xc6,0x04,0x65,0xd1]
@ CHECK: smultbne	r2, r3, r2      @ encoding: [0xa3,0x02,0x62,0x11]
@ CHECK: smultteq	r8, r3, r4      @ encoding: [0xe3,0x04,0x68,0x01]


@------------------------------------------------------------------------------
@ SMULL
@------------------------------------------------------------------------------
        smull r3, r9, r0, r1
        smulls r3, r9, r0, r2
        smulleq r8, r3, r4, r5
        smullseq r8, r3, r4, r3

@ CHECK: smull	r3, r9, r0, r1          @ encoding: [0x90,0x31,0xc9,0xe0]
@ CHECK: smulls	r3, r9, r0, r2          @ encoding: [0x90,0x32,0xd9,0xe0]
@ CHECK: smulleq	r8, r3, r4, r5  @ encoding: [0x94,0x85,0xc3,0x00]
@ CHECK: smullseq	r8, r3, r4, r3  @ encoding: [0x94,0x83,0xd3,0x00]


@------------------------------------------------------------------------------
@ SMULWB/SMULWT
@------------------------------------------------------------------------------
        smulwb r3, r9, r0
        smulwt r3, r9, r2

@ CHECK: smulwb	r3, r9, r0              @ encoding: [0xa9,0x00,0x23,0xe1]
@ CHECK: smulwt	r3, r9, r2              @ encoding: [0xe9,0x02,0x23,0xe1]


@------------------------------------------------------------------------------
@ SMUSD/SMUSDX
@------------------------------------------------------------------------------
        smusd r3, r0, r1
        smusdx r3, r9, r2
        smusdeq r8, r3, r2
        smusdxne r7, r4, r3

@ CHECK: smusd	r3, r0, r1              @ encoding: [0x50,0xf1,0x03,0xe7]
@ CHECK: smusdx	r3, r9, r2              @ encoding: [0x79,0xf2,0x03,0xe7]
@ CHECK: smusdeq	r8, r3, r2      @ encoding: [0x53,0xf2,0x08,0x07]
@ CHECK: smusdxne	r7, r4, r3      @ encoding: [0x74,0xf3,0x07,0x17]


@------------------------------------------------------------------------------
@ SRS
@------------------------------------------------------------------------------
        srsda sp, #5
        srsdb sp, #1
        srsia sp, #0
        srsib sp, #15

        srsda sp!, #31
        srsdb sp!, #19
        srsia sp!, #2
        srsib sp!, #14

        srsfa sp, #11
        srsea sp, #10
        srsfd sp, #9
        srsed sp, #5

        srsfa sp!, #5
        srsea sp!, #5
        srsfd sp!, #5
        srsed sp!, #5

        srs sp, #5
        srs sp!, #5

@ CHECK: srsda	sp, #5                  @ encoding: [0x05,0x05,0x4d,0xf8]
@ CHECK: srsdb	sp, #1                  @ encoding: [0x01,0x05,0x4d,0xf9]
@ CHECK: srsia	sp, #0                  @ encoding: [0x00,0x05,0xcd,0xf8]
@ CHECK: srsib	sp, #15                 @ encoding: [0x0f,0x05,0xcd,0xf9]

@ CHECK: srsda	sp!, #31                @ encoding: [0x1f,0x05,0x6d,0xf8]
@ CHECK: srsdb	sp!, #19                @ encoding: [0x13,0x05,0x6d,0xf9]
@ CHECK: srsia	sp!, #2                 @ encoding: [0x02,0x05,0xed,0xf8]
@ CHECK: srsib	sp!, #14                @ encoding: [0x0e,0x05,0xed,0xf9]

@ CHECK: srsib	sp, #11                 @ encoding: [0x0b,0x05,0xcd,0xf9]
@ CHECK: srsia	sp, #10                 @ encoding: [0x0a,0x05,0xcd,0xf8]
@ CHECK: srsdb	sp, #9                  @ encoding: [0x09,0x05,0x4d,0xf9]
@ CHECK: srsda	sp, #5                  @ encoding: [0x05,0x05,0x4d,0xf8]

@ CHECK: srsib	sp!, #5                 @ encoding: [0x05,0x05,0xed,0xf9]
@ CHECK: srsia	sp!, #5                 @ encoding: [0x05,0x05,0xed,0xf8]
@ CHECK: srsdb	sp!, #5                 @ encoding: [0x05,0x05,0x6d,0xf9]
@ CHECK: srsda	sp!, #5                 @ encoding: [0x05,0x05,0x6d,0xf8]

@ CHECK: srsia	sp, #5                  @ encoding: [0x05,0x05,0xcd,0xf8]
@ CHECK: srsia	sp!, #5                 @ encoding: [0x05,0x05,0xed,0xf8]

@ Compatibility aliases.
        srsda #5
        srsdb #1
        srsia #0
        srsib #15

        srsda #31!
        srsdb #19!
        srsia #2!
        srsib #14!

        srsfa #11
        srsea #10
        srsfd #9
        srsed #5

        srsfa #5!
        srsea #5!
        srsfd #5!
        srsed #5!

        srs #5
        srs #5!

@ CHECK: srsda	sp, #5                  @ encoding: [0x05,0x05,0x4d,0xf8]
@ CHECK: srsdb	sp, #1                  @ encoding: [0x01,0x05,0x4d,0xf9]
@ CHECK: srsia	sp, #0                  @ encoding: [0x00,0x05,0xcd,0xf8]
@ CHECK: srsib	sp, #15                 @ encoding: [0x0f,0x05,0xcd,0xf9]
@ CHECK: srsda	sp!, #31                @ encoding: [0x1f,0x05,0x6d,0xf8]
@ CHECK: srsdb	sp!, #19                @ encoding: [0x13,0x05,0x6d,0xf9]
@ CHECK: srsia	sp!, #2                 @ encoding: [0x02,0x05,0xed,0xf8]
@ CHECK: srsib	sp!, #14                @ encoding: [0x0e,0x05,0xed,0xf9]
@ CHECK: srsib	sp, #11                 @ encoding: [0x0b,0x05,0xcd,0xf9]
@ CHECK: srsia	sp, #10                 @ encoding: [0x0a,0x05,0xcd,0xf8]
@ CHECK: srsdb	sp, #9                  @ encoding: [0x09,0x05,0x4d,0xf9]
@ CHECK: srsda	sp, #5                  @ encoding: [0x05,0x05,0x4d,0xf8]
@ CHECK: srsib	sp!, #5                 @ encoding: [0x05,0x05,0xed,0xf9]
@ CHECK: srsia	sp!, #5                 @ encoding: [0x05,0x05,0xed,0xf8]
@ CHECK: srsdb	sp!, #5                 @ encoding: [0x05,0x05,0x6d,0xf9]
@ CHECK: srsda	sp!, #5                 @ encoding: [0x05,0x05,0x6d,0xf8]
@ CHECK: srsia	sp, #5                  @ encoding: [0x05,0x05,0xcd,0xf8]
@ CHECK: srsia	sp!, #5                 @ encoding: [0x05,0x05,0xed,0xf8]


@------------------------------------------------------------------------------
@ SSAT
@------------------------------------------------------------------------------
        ssat	r8, #1, r10
        ssat	r8, #1, r10, lsl #0
        ssat	r8, #1, r10, lsl #31
        ssat	r8, #1, r10, asr #32
        ssat	r8, #1, r10, asr #1

@ CHECK: ssat	r8, #1, r10             @ encoding: [0x1a,0x80,0xa0,0xe6]
@ CHECK: ssat	r8, #1, r10             @ encoding: [0x1a,0x80,0xa0,0xe6]
@ CHECK: ssat	r8, #1, r10, lsl #31    @ encoding: [0x9a,0x8f,0xa0,0xe6]
@ CHECK: ssat	r8, #1, r10, asr #32    @ encoding: [0x5a,0x80,0xa0,0xe6]
@ CHECK: ssat	r8, #1, r10, asr #1     @ encoding: [0xda,0x80,0xa0,0xe6]


@------------------------------------------------------------------------------
@ SSAT16
@------------------------------------------------------------------------------
        ssat16	r2, #1, r7
        ssat16	r3, #16, r5

@ CHECK: ssat16	r2, #1, r7              @ encoding: [0x37,0x2f,0xa0,0xe6]
@ CHECK: ssat16	r3, #16, r5             @ encoding: [0x35,0x3f,0xaf,0xe6]


@------------------------------------------------------------------------------
@ SSAX
@------------------------------------------------------------------------------
        ssax r2, r3, r4
        ssaxlt r2, r3, r4

@ CHECK: ssax	r2, r3, r4              @ encoding: [0x54,0x2f,0x13,0xe6]
@ CHECK: ssaxlt	r2, r3, r4              @ encoding: [0x54,0x2f,0x13,0xb6]

@------------------------------------------------------------------------------
@ SSUB16/SSUB8
@------------------------------------------------------------------------------
        ssub16 r1, r0, r6
        ssub16ne r5, r3, r2
        ssub8 r9, r2, r4
        ssub8eq r5, r1, r2

@ CHECK: ssub16	r1, r0, r6              @ encoding: [0x76,0x1f,0x10,0xe6]
@ CHECK: ssub16ne	r5, r3, r2      @ encoding: [0x72,0x5f,0x13,0x16]
@ CHECK: ssub8	r9, r2, r4              @ encoding: [0xf4,0x9f,0x12,0xe6]
@ CHECK: ssub8eq	r5, r1, r2      @ encoding: [0xf2,0x5f,0x11,0x06]

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

        stclo p12, c4, [r0, #4]
        stchi p13, c5, [r1]
        stccs p14, c6, [r2, #-224]
        stccc p15, c7, [r3, #-120]!
        stceq p5, c8, [r4], #16
        stcgt p4, c9, [r5], #-72
        stcllt p3, c10, [r6, #4]
        stclge p2, c11, [r7]
        stclle p1, c12, [r8, #-224]
        stclne p0, c13, [r9, #-120]!
        stcleq p6, c14, [r10], #16
        stclhi p7, c15, [r11], #-72

        stc2 p2, c8, [r1], { 25 }

@ CHECK: stc2	p0, c8, [r1, #4]        @ encoding: [0x01,0x80,0x81,0xfd]
@ CHECK: stc2	p1, c7, [r2]            @ encoding: [0x00,0x71,0x82,0xfd]
@ CHECK: stc2	p2, c6, [r3, #-224]     @ encoding: [0x38,0x62,0x03,0xfd]
@ CHECK: stc2	p3, c5, [r4, #-120]!    @ encoding: [0x1e,0x53,0x24,0xfd]
@ CHECK: stc2	p4, c4, [r5], #16       @ encoding: [0x04,0x44,0xa5,0xfc]
@ CHECK: stc2	p5, c3, [r6], #-72      @ encoding: [0x12,0x35,0x26,0xfc]
@ CHECK: stc2l	p6, c2, [r7, #4]        @ encoding: [0x01,0x26,0xc7,0xfd]
@ CHECK: stc2l	p7, c1, [r8]            @ encoding: [0x00,0x17,0xc8,0xfd]
@ CHECK: stc2l	p8, c0, [r9, #-224]     @ encoding: [0x38,0x08,0x49,0xfd]
@ CHECK: stc2l	p9, c1, [r10, #-120]!   @ encoding: [0x1e,0x19,0x6a,0xfd]
@ CHECK: stc2l	p0, c2, [r11], #16      @ encoding: [0x04,0x20,0xeb,0xfc]
@ CHECK: stc2l	p1, c3, [r12], #-72     @ encoding: [0x12,0x31,0x6c,0xfc]

@ CHECK: stc	p12, c4, [r0, #4]       @ encoding: [0x01,0x4c,0x80,0xed]
@ CHECK: stc	p13, c5, [r1]           @ encoding: [0x00,0x5d,0x81,0xed]
@ CHECK: stc	p14, c6, [r2, #-224]    @ encoding: [0x38,0x6e,0x02,0xed]
@ CHECK: stc	p15, c7, [r3, #-120]!   @ encoding: [0x1e,0x7f,0x23,0xed]
@ CHECK: stc	p5, c8, [r4], #16       @ encoding: [0x04,0x85,0xa4,0xec]
@ CHECK: stc	p4, c9, [r5], #-72      @ encoding: [0x12,0x94,0x25,0xec]
@ CHECK: stcl	p3, c10, [r6, #4]       @ encoding: [0x01,0xa3,0xc6,0xed]
@ CHECK: stcl	p2, c11, [r7]           @ encoding: [0x00,0xb2,0xc7,0xed]
@ CHECK: stcl	p1, c12, [r8, #-224]    @ encoding: [0x38,0xc1,0x48,0xed]
@ CHECK: stcl	p0, c13, [r9, #-120]!   @ encoding: [0x1e,0xd0,0x69,0xed]
@ CHECK: stcl	p6, c14, [r10], #16     @ encoding: [0x04,0xe6,0xea,0xec]
@ CHECK: stcl	p7, c15, [r11], #-72    @ encoding: [0x12,0xf7,0x6b,0xec]

@ CHECK: stclo	p12, c4, [r0, #4]       @ encoding: [0x01,0x4c,0x80,0x3d]
@ CHECK: stchi	p13, c5, [r1]           @ encoding: [0x00,0x5d,0x81,0x8d]
@ CHECK: stchs	p14, c6, [r2, #-224]    @ encoding: [0x38,0x6e,0x02,0x2d]
@ CHECK: stclo	p15, c7, [r3, #-120]!   @ encoding: [0x1e,0x7f,0x23,0x3d]
@ CHECK: stceq	p5, c8, [r4], #16       @ encoding: [0x04,0x85,0xa4,0x0c]
@ CHECK: stcgt	p4, c9, [r5], #-72      @ encoding: [0x12,0x94,0x25,0xcc]
@ CHECK: stcllt	p3, c10, [r6, #4]       @ encoding: [0x01,0xa3,0xc6,0xbd]
@ CHECK: stclge	p2, c11, [r7]           @ encoding: [0x00,0xb2,0xc7,0xad]
@ CHECK: stclle	p1, c12, [r8, #-224]    @ encoding: [0x38,0xc1,0x48,0xdd]
@ CHECK: stclne	p0, c13, [r9, #-120]!   @ encoding: [0x1e,0xd0,0x69,0x1d]
@ CHECK: stcleq	p6, c14, [r10], #16     @ encoding: [0x04,0xe6,0xea,0x0c]
@ CHECK: stclhi	p7, c15, [r11], #-72    @ encoding: [0x12,0xf7,0x6b,0x8c]

@ CHECK: stc2	p2, c8, [r1], {25}      @ encoding: [0x19,0x82,0x81,0xfc]


@------------------------------------------------------------------------------
@ STM*
@------------------------------------------------------------------------------
        stm       r2, {r1,r3-r6,sp}
        stmia     r3, {r1,r3-r6,lr}
        stmib     r4, {r1,r3-r6,sp}
        stmda     r5, {r1,r3-r6,sp}
        stmdb     r6, {r1,r3-r6,r8}
        stmfd     sp, {r1,r3-r6,sp}

        @ with update
        stm       r8!, {r1,r3-r6,sp}
        stmib     r9!, {r1,r3-r6,sp}
        stmda     sp!, {r1,r3-r6}
        stmdb     r0!, {r1,r5,r7,sp}

@ CHECK: stm	r2, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x82,0xe8]
@ CHECK: stm	r3, {r1, r3, r4, r5, r6, lr} @ encoding: [0x7a,0x40,0x83,0xe8]
@ CHECK: stmib	r4, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x84,0xe9]
@ CHECK: stmda	r5, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x05,0xe8]
@ CHECK: stmdb	r6, {r1, r3, r4, r5, r6, r8} @ encoding: [0x7a,0x01,0x06,0xe9]
@ CHECK: stmdb	sp, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0x0d,0xe9]

@ CHECK: stm	r8!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xa8,0xe8]
@ CHECK: stmib	r9!, {r1, r3, r4, r5, r6, sp} @ encoding: [0x7a,0x20,0xa9,0xe9]
@ CHECK: stmda	sp!, {r1, r3, r4, r5, r6}     @ encoding: [0x7a,0x00,0x2d,0xe8]
@ CHECK: stmdb	r0!, {r1, r5, r7, sp}         @ encoding: [0xa2,0x20,0x20,0xe9]


@------------------------------------------------------------------------------
@ STREX/STREXB/STREXH/STREXD
@------------------------------------------------------------------------------
        strexb  r1, r3, [r4]
        strexh  r4, r2, [r5]
        strex  r2, r1, [r7]
        strexd  r6, r2, r3, [r8]

@ CHECK: strexb	r1, r3, [r4]            @ encoding: [0x93,0x1f,0xc4,0xe1]
@ CHECK: strexh	r4, r2, [r5]            @ encoding: [0x92,0x4f,0xe5,0xe1]
@ CHECK: strex	r2, r1, [r7]            @ encoding: [0x91,0x2f,0x87,0xe1]
@ CHECK: strexd	r6, r2, r3, [r8]        @ encoding: [0x92,0x6f,0xa8,0xe1]

@------------------------------------------------------------------------------
@ STR
@------------------------------------------------------------------------------
        strpl	r3, [r10, #-0]!
        strpl	r3, [r10, #0]!

@ CHECK: strpl	r3, [r10, #-0]!         @ encoding: [0x00,0x30,0x2a,0x55]
@ CHECK: strpl	r3, [r10, #0]!          @ encoding: [0x00,0x30,0xaa,0x55]

@------------------------------------------------------------------------------
@ SUB
@------------------------------------------------------------------------------
        sub r4, r5, #0xf000
        sub r4, r5, $0xf000
        sub r4, r5, 0xf000
        sub r7, r8, #(0xff << 16)
        sub r7, r8, #-2147483638
        sub r7, r8, #42, #2
        sub r7, r8, #40, #2
        sub r7, r8, $40, $2
        sub r7, r8, 40, 2
        sub r7, r8, (20 * 2), (1 << 1)
        sub r4, r5, r6
        sub r4, r5, r6, lsl #5
        sub r4, r5, r6, lsr #5
        sub r4, r5, r6, lsr #5
        sub r4, r5, r6, asr #5
        sub r4, r5, r6, ror #5
        sub r6, r7, r8, lsl r9
        sub r6, r7, r8, lsr r9
        sub r6, r7, r8, asr r9
        sub r6, r7, r8, ror r9

        @ destination register is optional
        sub r5, #0xf000
        sub r5, $0xf000
        sub r5, 0xf000
        sub r7, #(0xff << 16)
        sub r7, #-2147483638
        sub r7, #42, #2
        sub r7, #40, #2
        sub r7, $40, $2
        sub r7, 40, 2
        sub r7, (20 * 2), (1 << 1)
        sub r4, r5
        sub r4, r5, lsl #5
        sub r4, r5, lsr #5
        sub r4, r5, lsr #5
        sub r4, r5, asr #5
        sub r4, r5, ror #5
        sub r6, r7, lsl r9
        sub r6, r7, lsr r9
        sub r6, r7, asr r9
        sub r6, r7, ror r9

@ CHECK: sub	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x45,0xe2]
@ CHECK: sub	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x45,0xe2]
@ CHECK: sub	r4, r5, #61440          @ encoding: [0x0f,0x4a,0x45,0xe2]
@ CHECK: sub	r7, r8, #16711680       @ encoding: [0xff,0x78,0x48,0xe2]
@ CHECK: sub    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x48,0xe2]
@ CHECK: sub    r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x48,0xe2]
@ CHECK: sub    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x48,0xe2]
@ CHECK: sub    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x48,0xe2]
@ CHECK: sub    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x48,0xe2]
@ CHECK: sub    r7, r8, #40, #2         @ encoding: [0x28,0x71,0x48,0xe2]
@ CHECK: sub	r4, r5, r6              @ encoding: [0x06,0x40,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, lsl #5      @ encoding: [0x86,0x42,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, lsr #5      @ encoding: [0xa6,0x42,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, asr #5      @ encoding: [0xc6,0x42,0x45,0xe0]
@ CHECK: sub	r4, r5, r6, ror #5      @ encoding: [0xe6,0x42,0x45,0xe0]
@ CHECK: sub	r6, r7, r8, lsl r9      @ encoding: [0x18,0x69,0x47,0xe0]
@ CHECK: sub	r6, r7, r8, lsr r9      @ encoding: [0x38,0x69,0x47,0xe0]
@ CHECK: sub	r6, r7, r8, asr r9      @ encoding: [0x58,0x69,0x47,0xe0]
@ CHECK: sub	r6, r7, r8, ror r9      @ encoding: [0x78,0x69,0x47,0xe0]


@ CHECK: sub	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x45,0xe2]
@ CHECK: sub	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x45,0xe2]
@ CHECK: sub	r5, r5, #61440          @ encoding: [0x0f,0x5a,0x45,0xe2]
@ CHECK: sub	r7, r7, #16711680       @ encoding: [0xff,0x78,0x47,0xe2]
@ CHECK: sub	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x47,0xe2]
@ CHECK: sub	r7, r7, #-2147483638    @ encoding: [0x2a,0x71,0x47,0xe2]
@ CHECK: sub	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x47,0xe2]
@ CHECK: sub	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x47,0xe2
@ CHECK: sub	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x47,0xe2]
@ CHECK: sub	r7, r7, #40, #2         @ encoding: [0x28,0x71,0x47,0xe2]
@ CHECK: sub	r4, r4, r5              @ encoding: [0x05,0x40,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, lsl #5      @ encoding: [0x85,0x42,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, lsr #5      @ encoding: [0xa5,0x42,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, asr #5      @ encoding: [0xc5,0x42,0x44,0xe0]
@ CHECK: sub	r4, r4, r5, ror #5      @ encoding: [0xe5,0x42,0x44,0xe0]
@ CHECK: sub	r6, r6, r7, lsl r9      @ encoding: [0x17,0x69,0x46,0xe0]
@ CHECK: sub	r6, r6, r7, lsr r9      @ encoding: [0x37,0x69,0x46,0xe0]
@ CHECK: sub	r6, r6, r7, asr r9      @ encoding: [0x57,0x69,0x46,0xe0]
@ CHECK: sub	r6, r6, r7, ror r9      @ encoding: [0x77,0x69,0x46,0xe0]

    @ Test right shift by 32, which is encoded as 0
    sub r3, r1, r2, lsr #32
    sub r3, r1, r2, asr #32
@ CHECK: sub	r3, r1, r2, lsr #32     @ encoding: [0x22,0x30,0x41,0xe0]
@ CHECK: sub	r3, r1, r2, asr #32     @ encoding: [0x42,0x30,0x41,0xe0]

@------------------------------------------------------------------------------
@ SUBS
@------------------------------------------------------------------------------
    subs r7, r8, #16711680
    subs r7, r8, $16711680
    subs r7, r8, 16711680
    subs r7, r8, #(0xff << 16)
    subs r7, r8, #-2147483638
    subs r7, r8, #42, #2
    subs r7, r8, #40, #2
    subs r7, r8, $40, $2
    subs r7, r8, 40, 2
    subs r7, r8, (20 * 2), (1 << 1)

@ CHECK: subs	r7, r8, #16711680       @ encoding: [0xff,0x78,0x58,0xe2]
@ CHECK: subs	r7, r8, #16711680       @ encoding: [0xff,0x78,0x58,0xe2]
@ CHECK: subs	r7, r8, #16711680       @ encoding: [0xff,0x78,0x58,0xe2]
@ CHECK: subs	r7, r8, #16711680       @ encoding: [0xff,0x78,0x58,0xe2]
@ CHECK: subs   r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x58,0xe2]
@ CHECK: subs   r7, r8, #-2147483638    @ encoding: [0x2a,0x71,0x58,0xe2]
@ CHECK: subs   r7, r8, #40, #2         @ encoding: [0x28,0x71,0x58,0xe2]
@ CHECK: subs   r7, r8, #40, #2         @ encoding: [0x28,0x71,0x58,0xe2]
@ CHECK: subs   r7, r8, #40, #2         @ encoding: [0x28,0x71,0x58,0xe2]
@ CHECK: subs   r7, r8, #40, #2         @ encoding: [0x28,0x71,0x58,0xe2]

@------------------------------------------------------------------------------
@ SVC
@------------------------------------------------------------------------------
        svc #16
        svc #0
        svc #0xffffff

@ CHECK: svc	#16                     @ encoding: [0x10,0x00,0x00,0xef]
@ CHECK: svc	#0                      @ encoding: [0x00,0x00,0x00,0xef]
@ CHECK: svc	#16777215               @ encoding: [0xff,0xff,0xff,0xef]


@------------------------------------------------------------------------------
@ SWP/SWPB
@------------------------------------------------------------------------------
        swp r1, r2, [r3]
        swp r4, r4, [r6]
        swpb r5, r1, [r9]

@ CHECK: swp	r1, r2, [r3]            @ encoding: [0x92,0x10,0x03,0xe1]
@ CHECK: swp	r4, r4, [r6]            @ encoding: [0x94,0x40,0x06,0xe1]
@ CHECK: swpb	r5, r1, [r9]            @ encoding: [0x91,0x50,0x49,0xe1]


@------------------------------------------------------------------------------
@ SXTAB
@------------------------------------------------------------------------------
        sxtab r2, r3, r4
        sxtab r4, r5, r6, ror #0
        sxtablt r6, r2, r9, ror #8
        sxtab r5, r1, r4, ror #16
        sxtab r7, r8, r3, ror #24

@ CHECK: sxtab	r2, r3, r4              @ encoding: [0x74,0x20,0xa3,0xe6]
@ CHECK: sxtab	r4, r5, r6              @ encoding: [0x76,0x40,0xa5,0xe6]
@ CHECK: sxtablt r6, r2, r9, ror #8     @ encoding: [0x79,0x64,0xa2,0xb6]
@ CHECK: sxtab	r5, r1, r4, ror #16     @ encoding: [0x74,0x58,0xa1,0xe6]
@ CHECK: sxtab	r7, r8, r3, ror #24     @ encoding: [0x73,0x7c,0xa8,0xe6]


@------------------------------------------------------------------------------
@ SXTAB16
@------------------------------------------------------------------------------
        sxtab16ge r0, r1, r4
        sxtab16 r6, r2, r7, ror #0
        sxtab16 r3, r5, r8, ror #8
        sxtab16 r3, r2, r1, ror #16
        sxtab16eq r1, r2, r3, ror #24

@ CHECK: sxtab16ge	r0, r1, r4      @ encoding: [0x74,0x00,0x81,0xa6]
@ CHECK: sxtab16	r6, r2, r7      @ encoding: [0x77,0x60,0x82,0xe6]
@ CHECK: sxtab16 r3, r5, r8, ror #8     @ encoding: [0x78,0x34,0x85,0xe6]
@ CHECK: sxtab16 r3, r2, r1, ror #16    @ encoding: [0x71,0x38,0x82,0xe6]
@ CHECK: sxtab16eq r1, r2, r3, ror #24  @ encoding: [0x73,0x1c,0x82,0x06]

@------------------------------------------------------------------------------
@ SXTAH
@------------------------------------------------------------------------------
        sxtah r1, r3, r9
        sxtahhi r6, r1, r6, ror #0
        sxtah r3, r8, r3, ror #8
        sxtahlo r2, r2, r4, ror #16
        sxtah r9, r3, r3, ror #24

@ CHECK: sxtah	r1, r3, r9              @ encoding: [0x79,0x10,0xb3,0xe6]
@ CHECK: sxtahhi	r6, r1, r6      @ encoding: [0x76,0x60,0xb1,0x86]
@ CHECK: sxtah	r3, r8, r3, ror #8      @ encoding: [0x73,0x34,0xb8,0xe6]
@ CHECK: sxtahlo r2, r2, r4, ror #16    @ encoding: [0x74,0x28,0xb2,0x36]
@ CHECK: sxtah	r9, r3, r3, ror #24     @ encoding: [0x73,0x9c,0xb3,0xe6]

@------------------------------------------------------------------------------
@ SXTB
@------------------------------------------------------------------------------
        sxtbge r2, r4
        sxtb r5, r6, ror #0
        sxtb r6, r9, ror #8
        sxtbcc r5, r1, ror #16
        sxtb r8, r3, ror #24

@ CHECK: sxtbge	r2, r4                  @ encoding: [0x74,0x20,0xaf,0xa6]
@ CHECK: sxtb	r5, r6                  @ encoding: [0x76,0x50,0xaf,0xe6]
@ CHECK: sxtb	r6, r9, ror #8          @ encoding: [0x79,0x64,0xaf,0xe6]
@ CHECK: sxtblo	r5, r1, ror #16         @ encoding: [0x71,0x58,0xaf,0x36]
@ CHECK: sxtb	r8, r3, ror #24         @ encoding: [0x73,0x8c,0xaf,0xe6]


@------------------------------------------------------------------------------
@ SXTB16
@------------------------------------------------------------------------------
        sxtb16 r1, r4
        sxtb16 r6, r7, ror #0
        sxtb16cs r3, r5, ror #8
        sxtb16 r3, r1, ror #16
        sxtb16ge r2, r3, ror #24

@ CHECK: sxtb16	r1, r4                  @ encoding: [0x74,0x10,0x8f,0xe6]
@ CHECK: sxtb16	r6, r7                  @ encoding: [0x77,0x60,0x8f,0xe6]
@ CHECK: sxtb16hs	r3, r5, ror #8  @ encoding: [0x75,0x34,0x8f,0x26]
@ CHECK: sxtb16	r3, r1, ror #16         @ encoding: [0x71,0x38,0x8f,0xe6]
@ CHECK: sxtb16ge	r2, r3, ror #24 @ encoding: [0x73,0x2c,0x8f,0xa6]


@------------------------------------------------------------------------------
@ SXTH
@------------------------------------------------------------------------------
        sxthne r3, r9
        sxth r1, r6, ror #0
        sxth r3, r8, ror #8
        sxthle r2, r2, ror #16
        sxth r9, r3, ror #24

@ CHECK: sxthne	r3, r9                  @ encoding: [0x79,0x30,0xbf,0x16]
@ CHECK: sxth	r1, r6                  @ encoding: [0x76,0x10,0xbf,0xe6]
@ CHECK: sxth	r3, r8, ror #8          @ encoding: [0x78,0x34,0xbf,0xe6]
@ CHECK: sxthle	r2, r2, ror #16         @ encoding: [0x72,0x28,0xbf,0xd6]
@ CHECK: sxth	r9, r3, ror #24         @ encoding: [0x73,0x9c,0xbf,0xe6]


@------------------------------------------------------------------------------
@ TEQ
@------------------------------------------------------------------------------
        teq r5, #0xf000
        teq r5, $0xf000
        teq r5, 0xf000
        teq r7, #(0xff << 16)
        teq r7, #-2147483638
        teq r7, #42, #2
        teq r7, #40, #2
        teq r7, $40, $2
        teq r7, 40, 2
        teq r7, (20 * 2), (1 << 1)
        teq r4, r5
        teq r4, r5, lsl #5
        teq r4, r5, lsr #5
        teq r4, r5, lsr #5
        teq r4, r5, asr #5
        teq r4, r5, ror #5
        teq r6, r7, lsl r9
        teq r6, r7, lsr r9
        teq r6, r7, asr r9
        teq r6, r7, ror r9

@ CHECK: teq	r5, #61440              @ encoding: [0x0f,0x0a,0x35,0xe3]
@ CHECK: teq	r5, #61440              @ encoding: [0x0f,0x0a,0x35,0xe3]
@ CHECK: teq	r5, #61440              @ encoding: [0x0f,0x0a,0x35,0xe3]
@ CHECK: teq	r7, #16711680           @ encoding: [0xff,0x08,0x37,0xe3]
@ CHECK: teq    r7, #-2147483638        @ encoding: [0x2a,0x01,0x37,0xe3]
@ CHECK: teq    r7, #-2147483638        @ encoding: [0x2a,0x01,0x37,0xe3]
@ CHECK: teq    r7, #40, #2             @ encoding: [0x28,0x01,0x37,0xe3]
@ CHECK: teq    r7, #40, #2             @ encoding: [0x28,0x01,0x37,0xe3]
@ CHECK: teq    r7, #40, #2             @ encoding: [0x28,0x01,0x37,0xe3]
@ CHECK: teq    r7, #40, #2             @ encoding: [0x28,0x01,0x37,0xe3]
@ CHECK: teq	r4, r5                  @ encoding: [0x05,0x00,0x34,0xe1]
@ CHECK: teq	r4, r5, lsl #5          @ encoding: [0x85,0x02,0x34,0xe1]
@ CHECK: teq	r4, r5, lsr #5          @ encoding: [0xa5,0x02,0x34,0xe1]
@ CHECK: teq	r4, r5, lsr #5          @ encoding: [0xa5,0x02,0x34,0xe1]
@ CHECK: teq	r4, r5, asr #5          @ encoding: [0xc5,0x02,0x34,0xe1]
@ CHECK: teq	r4, r5, ror #5          @ encoding: [0xe5,0x02,0x34,0xe1]
@ CHECK: teq	r6, r7, lsl r9          @ encoding: [0x17,0x09,0x36,0xe1]
@ CHECK: teq	r6, r7, lsr r9          @ encoding: [0x37,0x09,0x36,0xe1]
@ CHECK: teq	r6, r7, asr r9          @ encoding: [0x57,0x09,0x36,0xe1]
@ CHECK: teq	r6, r7, ror r9          @ encoding: [0x77,0x09,0x36,0xe1]


@------------------------------------------------------------------------------
@ TST
@------------------------------------------------------------------------------
        tst r5, #0xf000
        tst r5, $0xf000
        tst r5, 0xf000
        tst r7, #(0xff << 16)
        tst r7, #-2147483638
        tst r7, #42, #2
        tst r7, #40, #2
        tst r7, $40, $2
        tst r7, 40, 2
        tst r7, (20 * 2), (1 << 1)
        tst r4, r5
        tst r4, r5, lsl #5
        tst r4, r5, lsr #5
        tst r4, r5, lsr #5
        tst r4, r5, asr #5
        tst r4, r5, ror #5
        tst r6, r7, lsl r9
        tst r6, r7, lsr r9
        tst r6, r7, asr r9
        tst r6, r7, ror r9

@ CHECK: tst	r5, #61440              @ encoding: [0x0f,0x0a,0x15,0xe3]
@ CHECK: tst	r5, #61440              @ encoding: [0x0f,0x0a,0x15,0xe3]
@ CHECK: tst	r5, #61440              @ encoding: [0x0f,0x0a,0x15,0xe3]
@ CHECK: tst    r7, #16711680           @ encoding: [0xff,0x08,0x17,0xe3]
@ CHECK: tst    r7, #-2147483638        @ encoding: [0x2a,0x01,0x17,0xe3]
@ CHECK: tst    r7, #-2147483638        @ encoding: [0x2a,0x01,0x17,0xe3]
@ CHECK: tst    r7, #40, #2             @ encoding: [0x28,0x01,0x17,0xe3]
@ CHECK: tst    r7, #40, #2             @ encoding: [0x28,0x01,0x17,0xe3]
@ CHECK: tst    r7, #40, #2             @ encoding: [0x28,0x01,0x17,0xe3]
@ CHECK: tst    r7, #40, #2             @ encoding: [0x28,0x01,0x17,0xe3]
@ CHECK: tst	r4, r5                  @ encoding: [0x05,0x00,0x14,0xe1]
@ CHECK: tst	r4, r5, lsl #5          @ encoding: [0x85,0x02,0x14,0xe1]
@ CHECK: tst	r4, r5, lsr #5          @ encoding: [0xa5,0x02,0x14,0xe1]
@ CHECK: tst	r4, r5, lsr #5          @ encoding: [0xa5,0x02,0x14,0xe1]
@ CHECK: tst	r4, r5, asr #5          @ encoding: [0xc5,0x02,0x14,0xe1]
@ CHECK: tst	r4, r5, ror #5          @ encoding: [0xe5,0x02,0x14,0xe1]
@ CHECK: tst	r6, r7, lsl r9          @ encoding: [0x17,0x09,0x16,0xe1]
@ CHECK: tst	r6, r7, lsr r9          @ encoding: [0x37,0x09,0x16,0xe1]
@ CHECK: tst	r6, r7, asr r9          @ encoding: [0x57,0x09,0x16,0xe1]
@ CHECK: tst	r6, r7, ror r9          @ encoding: [0x77,0x09,0x16,0xe1]


@------------------------------------------------------------------------------
@ UADD16/UADD8
@------------------------------------------------------------------------------
        uadd16 r1, r2, r3
        uadd16gt r1, r2, r3
        uadd8 r1, r2, r3
        uadd8le r1, r2, r3

@ CHECK: uadd16	r1, r2, r3              @ encoding: [0x13,0x1f,0x52,0xe6]
@ CHECK: uadd16gt	r1, r2, r3      @ encoding: [0x13,0x1f,0x52,0xc6]
@ CHECK: uadd8	r1, r2, r3              @ encoding: [0x93,0x1f,0x52,0xe6]
@ CHECK: uadd8le r1, r2, r3             @ encoding: [0x93,0x1f,0x52,0xd6]


@------------------------------------------------------------------------------
@ UASX
@------------------------------------------------------------------------------
        uasx r9, r12, r0
        uasxeq r9, r12, r0

@ CHECK: uasx	r9, r12, r0             @ encoding: [0x30,0x9f,0x5c,0xe6]
@ CHECK: uasxeq	r9, r12, r0             @ encoding: [0x30,0x9f,0x5c,0x06]


@------------------------------------------------------------------------------
@ UBFX
@------------------------------------------------------------------------------
        ubfx r4, r5, #16, #1
        ubfxgt r4, r5, #16, #16

@ CHECK: ubfx	r4, r5, #16, #1         @ encoding: [0x55,0x48,0xe0,0xe7]
@ CHECK: ubfxgt	r4, r5, #16, #16        @ encoding: [0x55,0x48,0xef,0xc7]


@------------------------------------------------------------------------------
@ UHADD16/UHADD8
@------------------------------------------------------------------------------
        uhadd16 r4, r8, r2
        uhadd16gt r4, r8, r2
        uhadd8 r4, r8, r2
        uhadd8gt r4, r8, r2

@ CHECK: uhadd16	r4, r8, r2      @ encoding: [0x12,0x4f,0x78,0xe6]
@ CHECK: uhadd16gt	r4, r8, r2      @ encoding: [0x12,0x4f,0x78,0xc6]
@ CHECK: uhadd8	r4, r8, r2              @ encoding: [0x92,0x4f,0x78,0xe6]
@ CHECK: uhadd8gt	r4, r8, r2      @ encoding: [0x92,0x4f,0x78,0xc6]


@------------------------------------------------------------------------------
@ UHASX
@------------------------------------------------------------------------------
        uhasx r4, r8, r2
        uhasxgt r4, r8, r2

@ CHECK: uhasx	r4, r8, r2              @ encoding: [0x32,0x4f,0x78,0xe6]
@ CHECK: uhasxgt r4, r8, r2             @ encoding: [0x32,0x4f,0x78,0xc6]


@------------------------------------------------------------------------------
@ UHSUB16/UHSUB8
@------------------------------------------------------------------------------
        uhsub16 r4, r8, r2
        uhsub16gt r4, r8, r2
        uhsub8 r4, r8, r2
        uhsub8gt r4, r8, r2

@ CHECK: uhsub16	r4, r8, r2      @ encoding: [0x72,0x4f,0x78,0xe6]
@ CHECK: uhsub16gt	r4, r8, r2      @ encoding: [0x72,0x4f,0x78,0xc6]
@ CHECK: uhsub8	r4, r8, r2              @ encoding: [0xf2,0x4f,0x78,0xe6]
@ CHECK: uhsub8gt	r4, r8, r2      @ encoding: [0xf2,0x4f,0x78,0xc6]


@------------------------------------------------------------------------------
@ UMAAL
@------------------------------------------------------------------------------
        umaal r3, r4, r5, r6
        umaallt r3, r4, r5, r6

@ CHECK: umaal	r3, r4, r5, r6          @ encoding: [0x95,0x36,0x44,0xe0]
@ CHECK: umaallt	r3, r4, r5, r6  @ encoding: [0x95,0x36,0x44,0xb0]


@------------------------------------------------------------------------------
@ UMLAL
@------------------------------------------------------------------------------
        umlal r2, r4, r6, r8
        umlalgt r6, r1, r2, r6
        umlals r2, r9, r2, r3
        umlalseq r3, r5, r1, r2

@ CHECK: umlal	r2, r4, r6, r8          @ encoding: [0x96,0x28,0xa4,0xe0]
@ CHECK: umlalgt	r6, r1, r2, r6  @ encoding: [0x92,0x66,0xa1,0xc0]
@ CHECK: umlals	r2, r9, r2, r3          @ encoding: [0x92,0x23,0xb9,0xe0]
@ CHECK: umlalseq	r3, r5, r1, r2  @ encoding: [0x91,0x32,0xb5,0x00]


@------------------------------------------------------------------------------
@ UMULL
@------------------------------------------------------------------------------
        umull r2, r4, r6, r8
        umullgt r6, r1, r2, r6
        umulls r2, r9, r2, r3
        umullseq r3, r5, r1, r2

@ CHECK: umull	r2, r4, r6, r8          @ encoding: [0x96,0x28,0x84,0xe0]
@ CHECK: umullgt	r6, r1, r2, r6  @ encoding: [0x92,0x66,0x81,0xc0]
@ CHECK: umulls	r2, r9, r2, r3          @ encoding: [0x92,0x23,0x99,0xe0]
@ CHECK: umullseq	r3, r5, r1, r2  @ encoding: [0x91,0x32,0x95,0x00]


@------------------------------------------------------------------------------
@ UQADD16/UQADD8
@------------------------------------------------------------------------------
        uqadd16 r1, r2, r3
        uqadd16gt r4, r7, r9
        uqadd8 r3, r4, r8
        uqadd8le r8, r1, r2


@ CHECK: uqadd16	r1, r2, r3      @ encoding: [0x13,0x1f,0x62,0xe6]
@ CHECK: uqadd16gt	r4, r7, r9      @ encoding: [0x19,0x4f,0x67,0xc6]
@ CHECK: uqadd8	r3, r4, r8              @ encoding: [0x98,0x3f,0x64,0xe6]
@ CHECK: uqadd8le	r8, r1, r2      @ encoding: [0x92,0x8f,0x61,0xd6]


@------------------------------------------------------------------------------
@ UQASX
@------------------------------------------------------------------------------
        uqasx r2, r4, r1
        uqasxhi r5, r2, r9

@ CHECK: uqasx	r2, r4, r1              @ encoding: [0x31,0x2f,0x64,0xe6]
@ CHECK: uqasxhi	r5, r2, r9      @ encoding: [0x39,0x5f,0x62,0x86]


@------------------------------------------------------------------------------
@ UQSAX
@------------------------------------------------------------------------------
        uqsax r1, r3, r7
        uqsaxal r3, r6, r2

@ CHECK: uqsax	r1, r3, r7              @ encoding: [0x57,0x1f,0x63,0xe6]
@ CHECK: uqsax	r3, r6, r2              @ encoding: [0x52,0x3f,0x66,0xe6]


@------------------------------------------------------------------------------
@ UQSUB16/UQSUB8
@------------------------------------------------------------------------------
        uqsub16 r1, r5, r3
        uqsub16gt r3, r2, r5
        uqsub8 r2, r1, r4
        uqsub8le r4, r6, r9

@ CHECK: uqsub16	r1, r5, r3      @ encoding: [0x73,0x1f,0x65,0xe6]
@ CHECK: uqsub16gt	r3, r2, r5      @ encoding: [0x75,0x3f,0x62,0xc6]
@ CHECK: uqsub8	r2, r1, r4              @ encoding: [0xf4,0x2f,0x61,0xe6]
@ CHECK: uqsub8le	r4, r6, r9      @ encoding: [0xf9,0x4f,0x66,0xd6]


@------------------------------------------------------------------------------
@ USADA8/USAD8
@------------------------------------------------------------------------------
        usad8 r2, r1, r4
        usad8le r4, r6, r9
        usada8 r1, r5, r3, r7
        usada8gt r3, r2, r5, r1

@ CHECK: usad8	r2, r1, r4              @ encoding: [0x11,0xf4,0x82,0xe7]
@ CHECK: usad8le	r4, r6, r9      @ encoding: [0x16,0xf9,0x84,0xd7]
@ CHECK: usada8	r1, r5, r3, r7          @ encoding: [0x15,0x73,0x81,0xe7]
@ CHECK: usada8gt	r3, r2, r5, r1  @ encoding: [0x12,0x15,0x83,0xc7]


@------------------------------------------------------------------------------
@ USAT
@------------------------------------------------------------------------------
        usat	r8, #1, r10
        usat	r8, #4, r10, lsl #0
        usat	r8, #5, r10, lsl #31
        usat	r8, #31, r10, asr #32
        usat	r8, #16, r10, asr #1

@ CHECK: usat	r8, #1, r10             @ encoding: [0x1a,0x80,0xe1,0xe6]
@ CHECK: usat	r8, #4, r10             @ encoding: [0x1a,0x80,0xe4,0xe6]
@ CHECK: usat	r8, #5, r10, lsl #31    @ encoding: [0x9a,0x8f,0xe5,0xe6]
@ CHECK: usat	r8, #31, r10, asr #32   @ encoding: [0x5a,0x80,0xff,0xe6]
@ CHECK: usat	r8, #16, r10, asr #1    @ encoding: [0xda,0x80,0xf0,0xe6]


@------------------------------------------------------------------------------
@ USAT16
@------------------------------------------------------------------------------
        usat16	r2, #2, r7
        usat16	r3, #15, r5

@ CHECK: usat16	r2, #2, r7              @ encoding: [0x37,0x2f,0xe2,0xe6]
@ CHECK: usat16	r3, #15, r5             @ encoding: [0x35,0x3f,0xef,0xe6]


@------------------------------------------------------------------------------
@ USAX
@------------------------------------------------------------------------------
        usax r2, r3, r4
        usaxne r2, r3, r4

@ CHECK: usax	r2, r3, r4              @ encoding: [0x54,0x2f,0x53,0xe6]
@ CHECK: usaxne	r2, r3, r4              @ encoding: [0x54,0x2f,0x53,0x16]

@------------------------------------------------------------------------------
@ USUB16/USUB8
@------------------------------------------------------------------------------
        usub16 r4, r2, r7
        usub16hi r1, r1, r3
        usub8 r1, r8, r5
        usub8le r9, r2, r3

@ CHECK: usub16	r4, r2, r7              @ encoding: [0x77,0x4f,0x52,0xe6]
@ CHECK: usub16hi	r1, r1, r3      @ encoding: [0x73,0x1f,0x51,0x86]
@ CHECK: usub8	r1, r8, r5              @ encoding: [0xf5,0x1f,0x58,0xe6]
@ CHECK: usub8le	r9, r2, r3      @ encoding: [0xf3,0x9f,0x52,0xd6]


@------------------------------------------------------------------------------
@ UXTAB
@------------------------------------------------------------------------------
        uxtab r2, r3, r4
        uxtab r4, r5, r6, ror #0
        uxtablt r6, r2, r9, ror #8
        uxtab r5, r1, r4, ror #16
        uxtab r7, r8, r3, ror #24

@ CHECK: uxtab	r2, r3, r4              @ encoding: [0x74,0x20,0xe3,0xe6]
@ CHECK: uxtab	r4, r5, r6              @ encoding: [0x76,0x40,0xe5,0xe6]
@ CHECK: uxtablt r6, r2, r9, ror #8     @ encoding: [0x79,0x64,0xe2,0xb6]
@ CHECK: uxtab	r5, r1, r4, ror #16     @ encoding: [0x74,0x58,0xe1,0xe6]
@ CHECK: uxtab	r7, r8, r3, ror #24     @ encoding: [0x73,0x7c,0xe8,0xe6]


@------------------------------------------------------------------------------
@ UXTAB16
@------------------------------------------------------------------------------
        uxtab16ge r0, r1, r4
        uxtab16 r6, r2, r7, ror #0
        uxtab16 r3, r5, r8, ror #8
        uxtab16 r3, r2, r1, ror #16
        uxtab16eq r1, r2, r3, ror #24

@ CHECK: uxtab16ge	r0, r1, r4      @ encoding: [0x74,0x00,0xc1,0xa6]
@ CHECK: uxtab16	r6, r2, r7      @ encoding: [0x77,0x60,0xc2,0xe6]
@ CHECK: uxtab16	r3, r5, r8, ror #8 @ encoding: [0x78,0x34,0xc5,0xe6]
@ CHECK: uxtab16	r3, r2, r1, ror #16 @ encoding: [0x71,0x38,0xc2,0xe6]
@ CHECK: uxtab16eq	r1, r2, r3, ror #24 @ encoding: [0x73,0x1c,0xc2,0x06]


@------------------------------------------------------------------------------
@ UXTAH
@------------------------------------------------------------------------------
        uxtah r1, r3, r9
        uxtahhi r6, r1, r6, ror #0
        uxtah r3, r8, r3, ror #8
        uxtahlo r2, r2, r4, ror #16
        uxtah r9, r3, r3, ror #24

@ CHECK: uxtah	r1, r3, r9              @ encoding: [0x79,0x10,0xf3,0xe6]
@ CHECK: uxtahhi	r6, r1, r6      @ encoding: [0x76,0x60,0xf1,0x86]
@ CHECK: uxtah	r3, r8, r3, ror #8      @ encoding: [0x73,0x34,0xf8,0xe6]
@ CHECK: uxtahlo	r2, r2, r4, ror #16 @ encoding: [0x74,0x28,0xf2,0x36]
@ CHECK: uxtah	r9, r3, r3, ror #24     @ encoding: [0x73,0x9c,0xf3,0xe6]

@------------------------------------------------------------------------------
@ UXTB
@------------------------------------------------------------------------------
        uxtbge r2, r4
        uxtb r5, r6, ror #0
        uxtb r6, r9, ror #8
        uxtbcc r5, r1, ror #16
        uxtb r8, r3, ror #24

@ CHECK: uxtbge	r2, r4                  @ encoding: [0x74,0x20,0xef,0xa6]
@ CHECK: uxtb	r5, r6                  @ encoding: [0x76,0x50,0xef,0xe6]
@ CHECK: uxtb	r6, r9, ror #8          @ encoding: [0x79,0x64,0xef,0xe6]
@ CHECK: uxtblo	r5, r1, ror #16         @ encoding: [0x71,0x58,0xef,0x36]
@ CHECK: uxtb	r8, r3, ror #24         @ encoding: [0x73,0x8c,0xef,0xe6]


@------------------------------------------------------------------------------
@ UXTB16
@------------------------------------------------------------------------------
        uxtb16 r1, r4
        uxtb16 r6, r7, ror #0
        uxtb16cs r3, r5, ror #8
        uxtb16 r3, r1, ror #16
        uxtb16ge r2, r3, ror #24

@ CHECK: uxtb16	r1, r4                  @ encoding: [0x74,0x10,0xcf,0xe6]
@ CHECK: uxtb16	r6, r7                  @ encoding: [0x77,0x60,0xcf,0xe6]
@ CHECK: uxtb16hs	r3, r5, ror #8  @ encoding: [0x75,0x34,0xcf,0x26]
@ CHECK: uxtb16	r3, r1, ror #16         @ encoding: [0x71,0x38,0xcf,0xe6]
@ CHECK: uxtb16ge	r2, r3, ror #24 @ encoding: [0x73,0x2c,0xcf,0xa6]


@------------------------------------------------------------------------------
@ UXTH
@------------------------------------------------------------------------------
        uxthne r3, r9
        uxth r1, r6, ror #0
        uxth r3, r8, ror #8
        uxthle r2, r2, ror #16
        uxth r9, r3, ror #24

@ CHECK: uxthne	r3, r9                  @ encoding: [0x79,0x30,0xff,0x16]
@ CHECK: uxth	r1, r6                  @ encoding: [0x76,0x10,0xff,0xe6]
@ CHECK: uxth	r3, r8, ror #8          @ encoding: [0x78,0x34,0xff,0xe6]
@ CHECK: uxthle	r2, r2, ror #16         @ encoding: [0x72,0x28,0xff,0xd6]
@ CHECK: uxth	r9, r3, ror #24         @ encoding: [0x73,0x9c,0xff,0xe6]


@------------------------------------------------------------------------------
@ WFE/WFI/YIELD
@------------------------------------------------------------------------------
        wfe
        wfehi
        wfi
        wfilt
        yield
        yieldne
        hint #4
        hint #3
        hint #2
        hint #1
        hint #0
        hintgt #239

@ CHECK: wfe                            @ encoding: [0x02,0xf0,0x20,0xe3]
@ CHECK: wfehi                          @ encoding: [0x02,0xf0,0x20,0x83]
@ CHECK: wfi                            @ encoding: [0x03,0xf0,0x20,0xe3]
@ CHECK: wfilt                          @ encoding: [0x03,0xf0,0x20,0xb3]
@ CHECK: yield                          @ encoding: [0x01,0xf0,0x20,0xe3]
@ CHECK: yieldne                        @ encoding: [0x01,0xf0,0x20,0x13]
@ CHECK: sev                            @ encoding: [0x04,0xf0,0x20,0xe3]
@ CHECK: wfi                            @ encoding: [0x03,0xf0,0x20,0xe3]
@ CHECK: wfe                            @ encoding: [0x02,0xf0,0x20,0xe3]
@ CHECK: yield                          @ encoding: [0x01,0xf0,0x20,0xe3]
@ CHECK: nop                            @ encoding: [0x00,0xf0,0x20,0xe3]
@ CHECK: hintgt #239                    @ encoding: [0xef,0xf0,0x20,0xc3]
