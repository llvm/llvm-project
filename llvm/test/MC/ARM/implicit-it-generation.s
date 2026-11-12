@ RUN: llvm-mc -triple thumbv7a--none-eabi -arm-implicit-it=always < %s -show-encoding | FileCheck %s

@ Single instruction
  .section test1
@ CHECK-LABEL: test1
  addeq r0, #1
@ CHECK: it eq
@ CHECK: addeq

@ Multiple instructions, same condition
  .section test2
@ CHECK-LABEL: test2
  addeq r0, #1
  addeq r0, #1
  addeq r0, #1
  addeq r0, #1
@ CHECK: itttt eq
@ CHECK: addeq
@ CHECK: addeq
@ CHECK: addeq
@ CHECK: addeq

@ Multiple instructions, equal but opposite conditions
  .section test3
@ CHECK-LABEL: test3
  addeq r0, #1
  addne r0, #1
  addeq r0, #1
  addne r0, #1
@ CHECK: itete eq
@ CHECK: addeq
@ CHECK: addne
@ CHECK: addeq
@ CHECK: addne

@ Multiple instructions, unrelated conditions
  .section test4
@ CHECK-LABEL: test4
  addeq r0, #1
  addlt r0, #1
  addeq r0, #1
  addge r0, #1
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it lt
@ CHECK: addlt
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it ge
@ CHECK: addge

@ More than 4 instructions eligible for a block
  .section test5
@ CHECK-LABEL: test5
  addeq r0, #1
  addeq r0, #1
  addeq r0, #1
  addeq r0, #1
  addeq r0, #1
  addeq r0, #1
@ CHECK: itttt eq
@ CHECK: addeq
@ CHECK: addeq
@ CHECK: addeq
@ CHECK: addeq
@ CHECK: itt eq
@ CHECK: addeq
@ CHECK: addeq

@ Flush on a label
  .section test6
@ CHECK-LABEL: test6
  addeq r0, #1
label:
  addeq r0, #1
five:
  addeq r0, #1
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: label
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: five
@ CHECK: it eq
@ CHECK: addeq

@ Flush on a section-change directive
  .section test7a
@ CHECK-LABEL: test7a
  addeq r0, #1
  .section test7b
  addeq r0, #1
  .previous
  addeq r0, #1
  .pushsection test7c
  addeq r0, #1
  .popsection
  addeq r0, #1
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq

@ Flush on an ISA change (even to the same ISA)
  .section test8
@ CHECK-LABEL: test8
  addeq r0, #1
  .thumb
  addeq r0, #1
  .arm
  addeq r0, #1
  .thumb
  addeq r0, #1
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq

@ Flush on an arch, cpu or fpu change
  .section test9
@ CHECK-LABEL: test9
  addeq r0, #1
  .arch armv7-a
  addeq r0, #1
  .cpu cortex-a15
  addeq r0, #1
  .fpu vfpv3
  addeq r0, #1
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq

@ Flush on an unpredicable instruction
  .section test10
@ CHECK-LABEL: test10
  addeq r0, #1
  setend le
  addeq r0, #1
  hvc #0
  addeq r0, #1
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: setend le
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: hvc.w #0
@ CHECK: it eq
@ CHECK: addeq

@ Flush when reaching an explicit IT instruction
  .section test11
@ CHECK-LABEL: test11
  addeq r0, #1
  it eq
  addeq r0, #1
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq

@ Don't extend an explicit IT instruction
  .section test12
@ CHECK-LABEL: test12
  it eq
  addeq r0, #1
  addeq r0, #1
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq

@ Branch-like instructions can only be used at the end of an IT block, so
@ terminate it.
  .section test13
@ CHECK-LABEL: test13
  .cpu cortex-a15
  addeq pc, r0
  addeq pc, sp, pc
  ldreq pc, [r0, #4]
  ldreq pc, [r0, #-4]
  ldreq pc, [r0, r1]
  ldreq pc, [pc, #-0]
  moveq pc, r0
  bleq #4
  blxeq #4
  blxeq r0
  bxeq r0
  bxjeq r0
  tbbeq [r0, r1]
  tbheq [r0, r1, lsl #1]
  ereteq
  rfeiaeq r0
  rfeiaeq r0!
  rfedbeq r0
  rfedbeq r0!
  smceq #0
  ldmiaeq r0, {pc}
  ldmiaeq r0!, {r1, pc}
  ldmdbeq r0, {pc}
  ldmdbeq r0!, {r1, pc}
  popeq {pc}
  .arch armv8-m.main
  bxnseq r0
  blxnseq r0
@ CHECK: it eq
@ CHECK: addeq pc, r0
@ CHECK: it eq
@ CHECK: addeq pc, sp, pc
@ CHECK: it eq
@ CHECK: ldreq.w pc, [r0, #4]
@ CHECK: it eq
@ CHECK: ldreq pc, [r0, #-4]
@ CHECK: it eq
@ CHECK: ldreq.w pc, [r0, r1]
@ CHECK: it eq
@ CHECK: ldreq.w pc, [pc, #-0]
@ CHECK: it eq
@ CHECK: moveq pc, r0
@ CHECK: it eq
@ CHECK: bleq #4
@ CHECK: it eq
@ CHECK: blxeq #4
@ CHECK: it eq
@ CHECK: blxeq r0
@ CHECK: it eq
@ CHECK: bxeq r0
@ CHECK: it eq
@ CHECK: bxjeq r0
@ CHECK: it eq
@ CHECK: tbbeq [r0, r1]
@ CHECK: it eq
@ CHECK: tbheq [r0, r1, lsl #1]
@ CHECK: it eq
@ CHECK: ereteq
@ CHECK: it eq
@ CHECK: rfeiaeq r0
@ CHECK: it eq
@ CHECK: rfeiaeq r0!
@ CHECK: it eq
@ CHECK: rfedbeq r0
@ CHECK: it eq
@ CHECK: rfedbeq r0!
@ CHECK: it eq
@ CHECK: smceq #0
@ CHECK: it eq
@ CHECK: ldmeq.w r0, {pc}
@ CHECK: it eq
@ CHECK: ldmeq.w r0!, {r1, pc}
@ CHECK: it eq
@ CHECK: ldmdbeq r0, {pc}
@ CHECK: it eq
@ CHECK: ldmdbeq r0!, {r1, pc}
@ CHECK: it eq
@ CHECK: popeq {pc}
@ CHECK: it eq
@ CHECK: bxnseq r0
@ CHECK: it eq
@ CHECK: blxnseq r0

@ Thumb 16-bit ALU instructions set the flags iff they are not in an IT block,
@ so instruction matching must change when generating an implicit IT block.
  .section test14
@ CHECK-LABEL: test14
@ Outside an IT block, the 16-bit encoding must set flags
  add r0, #1
@ CHECK:add.w   r0, r0, #1              @ encoding: [0x00,0xf1,0x01,0x00] 
  adds r0, #1
@ CHECK: adds    r0, #1                  @ encoding: [0x01,0x30]
@ Inside an IT block, the 16-bit encoding can not set flags
  addeq r0, #1
@ CHECK: itt eq
@ CHECK: addeq   r0, #1                  @ encoding: [0x01,0x30]
  addseq r0, #1
@ CHECK: addseq.w        r0, r0, #1      @ encoding: [0x10,0xf1,0x01,0x00]

@ Some variants of the B instruction have their own condition code field, and
@ are not valid in IT blocks.
  .section test15
@ CHECK-LABEL: test15
@ Outside of an IT block, the 4 variants (narrow/wide,
@ predicated/non-predicated) are selected as normal, and the predicated
@ encodings are used instead of opening a new IT block:
  b #0x100
@ CHECK: b       #256                    @ encoding: [0x80,0xe0] 
  b #0x800
@ CHECK: b.w     #2048                   @ encoding: [0x00,0xf0,0x00,0xbc]
  beq #0x4
@ CHECK-NOT: it
@ CHECK: beq     #4                      @ encoding: [0x02,0xd0]
  beq #0x100
@ CHECK-NOT: it
@ CHECK: beq.w   #256                    @ encoding: [0x00,0xf0,0x80,0x80]

@ We could support "beq #0x100000" to "beq #0x1fffffc" by using t2Bcc in
@ an IT block (these currently fail as the target is out of range). However, long
@ ranges like this are rarely assembly-time constants, so this probably isn't
@ worth doing.

@ If we already have an open IT block, we can use the non-predicated encodings,
@ which have a greater range:
  addeq r0, r1
  beq #0x4
@ CHECK: itt eq
@ CHECK: addeq r0, r1
@ CHECK: beq     #4                      @ encoding: [0x02,0xe0]
  addeq r0, r1
  beq #0x100
@ CHECK: itt eq
@ CHECK: addeq r0, r1
@ CHECK: beq     #256                    @ encoding: [0x80,0xe0]
  addeq r0, r1
  beq #0x800
@ CHECK: itt eq
@ CHECK: addeq r0, r1
@ CHECK: beq.w   #2048                   @ encoding: [0x00,0xf0,0x00,0xbc]

@ If we have an open but incompatible IT block, we close it and use the
@ self-predicated encodings, without an IT block:
  addeq r0, r1
  bgt #0x4
@ CHECK: it eq
@ CHECK: addeq r0, r1
@ CHECK: bgt     #4                      @ encoding: [0x02,0xdc]
  addeq r0, r1
  bgt #0x100
@ CHECK: it eq
@ CHECK: addeq r0, r1
@ CHECK: bgt.w   #256                    @ encoding: [0x00,0xf3,0x80,0x80]

@ Breakpoint instructions are allowed in IT blocks, but are always executed
@ regardless of the condition flags. We could continue an IT block through
@ them, but currently do not.
  .section test16
@ CHECK-LABEL: test16
  addeq r0, r1
  bkpt #0
  addeq r0, r1
@ CHECK: it eq
@ CHECK: addeq r0, r1
@ CHECK: bkpt #0
@ CHECK: it eq
@ CHECK: addeq r0, r1

@ The .if directive causes entire assembly statments to be dropped before they
@ reach the IT block generation code. This happens to be exactly what we want,
@ and allows IT blocks to extend into and out of .if blocks. Only one arm of the
@ .if will be seen by the IT state tracking code, so the subeq shouldn't have
@ any effect here.
  .section test17
@ CHECK-LABEL: test17
  addeq r0, r1
  .if 1
  addeq r0, r1
  .else
  subeq r0, r1
  .endif
  addeq r0, r1
@ CHECK: ittt eq
@ CHECK: addeq
@ CHECK: addeq
@ CHECK: addeq

@ TODO: There are some other directives which we could continue through, such
@ as .set and .global, but we currently conservatively flush the IT block before
@ every directive (except for .if and friends, which are handled separately).
  .section test18
@ CHECK-LABEL: test18
  addeq r0, r1
  .set s, 1
  addeq r0, r1
@ CHECK: it eq
@ CHECK: addeq
@ CHECK: it eq
@ CHECK: addeq

@ The .rept directive can be used to create long IT blocks.
  .section test19
@ CHECK-LABEL: test19
  .rept 3
  addeq r0, r1
  subne r0, r1
  .endr
@ CHECK: itete eq
@ CHECK:  addeq r0, r1
@ CHECK:  subne r0, r0, r1
@ CHECK:  addeq r0, r1
@ CHECK:  subne r0, r0, r1
@ CHECK: ite eq
@ CHECK:  addeq r0, r1
@ CHECK:  subne r0, r0, r1

@ Flush at end of file
  .section test99
@ CHECK-LABEL: test99
  addeq r0, #1
@ CHECK: it eq
@ CHECK: addeq
