@ PR28647
@ RUN: llvm-mc < %s -triple=thumbv7a-linux-gnueabi -filetype=obj -o - \
@ RUN: | llvm-objdump -d --triple=thumbv7a-linux-gnueabi - | FileCheck %s
    .text
    .syntax unified
    .balign 2
@ Thumb2 modified immediate instructions
    add r1,r1, sym0
    sub r1,r2, sym1
    cmp r2,    sym2
    and r4,r4, sym3
    orr r8,r9, sym4
    teq r1,    sym5
    tst r1,    sym6
    sbc r1,r1, sym7
    adc r1,r0, sym8
@CHECK: add.w   r1, r1, #255
@CHECK: sub.w   r1, r2, #16711935
@CHECK: cmp.w   r2, #4278255360
@CHECK: and     r4, r4, #303174162
@CHECK: orr     r8, r9, #2852126720
@CHECK: teq.w   r1, #1426063360
@CHECK: tst.w   r1, #713031680
@CHECK: sbc     r1, r1, #2785280
@CHECK: adc     r1, r0, #340

.L1:
    sub r3, r3, #.L2 - .L1
.L2:
@CHECK: sub.w   r3, r3, #4

@ mov without :upper16: or :lower16: should match mov with modified immediate
     mov r1, sym3
@CHECK: mov.w   r1, #303174162

@ Modified immediate constants
    .equ sym0, 0x000000ff
    .equ sym1, 0x00ff00ff
    .equ sym2, 0xff00ff00
    .equ sym3, 0x12121212
    .equ sym4, 0xaa000000
    .equ sym5, 0x55000000
    .equ sym6, 0x2a800000
    .equ sym7, 0x002a8000
    .equ sym8, 0x00000154
