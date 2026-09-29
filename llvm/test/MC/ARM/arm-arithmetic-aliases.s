@ RUN: llvm-mc -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

foo:
@ CHECK: foo

sub r2, r2, #6
sub r2, #6
sub r2, r2, r3
sub r2, r3

@ CHECK: sub r2, r2, #6              @ encoding: [0x06,0x20,0x42,0xe2]
@ CHECK: sub r2, r2, #6              @ encoding: [0x06,0x20,0x42,0xe2]
@ CHECK: sub r2, r2, r3              @ encoding: [0x03,0x20,0x42,0xe0]
@ CHECK: sub r2, r2, r3              @ encoding: [0x03,0x20,0x42,0xe0]

add r2, r2, #6
add r2, #6
add r2, r2, r3
add r2, r3

@ CHECK: add r2, r2, #6              @ encoding: [0x06,0x20,0x82,0xe2]
@ CHECK: add r2, r2, #6              @ encoding: [0x06,0x20,0x82,0xe2]
@ CHECK: add r2, r2, r3              @ encoding: [0x03,0x20,0x82,0xe0]
@ CHECK: add r2, r2, r3              @ encoding: [0x03,0x20,0x82,0xe0]

and r2, r2, #6
and r2, #6
and r2, r2, r3
and r2, r3

@ CHECK: and r2, r2, #6              @ encoding: [0x06,0x20,0x02,0xe2]
@ CHECK: and r2, r2, #6              @ encoding: [0x06,0x20,0x02,0xe2]
@ CHECK: and r2, r2, r3              @ encoding: [0x03,0x20,0x02,0xe0]
@ CHECK: and r2, r2, r3              @ encoding: [0x03,0x20,0x02,0xe0]

orr r2, r2, #6
orr r2, #6
orr r2, r2, r3
orr r2, r3

@ CHECK: orr r2, r2, #6              @ encoding: [0x06,0x20,0x82,0xe3]
@ CHECK: orr r2, r2, #6              @ encoding: [0x06,0x20,0x82,0xe3]
@ CHECK: orr r2, r2, r3              @ encoding: [0x03,0x20,0x82,0xe1]
@ CHECK: orr r2, r2, r3              @ encoding: [0x03,0x20,0x82,0xe1]

eor r2, r2, #6
eor r2, #6
eor r2, r2, r3
eor r2, r3

@ CHECK: eor r2, r2, #6              @ encoding: [0x06,0x20,0x22,0xe2]
@ CHECK: eor r2, r2, #6              @ encoding: [0x06,0x20,0x22,0xe2]
@ CHECK: eor r2, r2, r3              @ encoding: [0x03,0x20,0x22,0xe0]
@ CHECK: eor r2, r2, r3              @ encoding: [0x03,0x20,0x22,0xe0]

bic r2, r2, #6
bic r2, #6
bic r2, r2, r3
bic r2, r3

@ CHECK: bic r2, r2, #6              @ encoding: [0x06,0x20,0xc2,0xe3]
@ CHECK: bic r2, r2, #6              @ encoding: [0x06,0x20,0xc2,0xe3]
@ CHECK: bic r2, r2, r3              @ encoding: [0x03,0x20,0xc2,0xe1]
@ CHECK: bic r2, r2, r3              @ encoding: [0x03,0x20,0xc2,0xe1]


@ Also check that we handle the predicate and cc_out operands.
subseq r2, r2, #6
subseq r2, #6
subseq r2, r2, r3
subseq r2, r3

@ CHECK: subseq r2, r2, #6              @ encoding: [0x06,0x20,0x52,0x02]
@ CHECK: subseq r2, r2, #6              @ encoding: [0x06,0x20,0x52,0x02]
@ CHECK: subseq r2, r2, r3              @ encoding: [0x03,0x20,0x52,0x00]
@ CHECK: subseq r2, r2, r3              @ encoding: [0x03,0x20,0x52,0x00]

addseq r2, r2, #6
addseq r2, #6
addseq r2, r2, r3
addseq r2, r3

@ CHECK: addseq r2, r2, #6              @ encoding: [0x06,0x20,0x92,0x02]
@ CHECK: addseq r2, r2, #6              @ encoding: [0x06,0x20,0x92,0x02]
@ CHECK: addseq r2, r2, r3              @ encoding: [0x03,0x20,0x92,0x00]
@ CHECK: addseq r2, r2, r3              @ encoding: [0x03,0x20,0x92,0x00]

andseq r2, r2, #6
andseq r2, #6
andseq r2, r2, r3
andseq r2, r3

@ CHECK: andseq r2, r2, #6              @ encoding: [0x06,0x20,0x12,0x02]
@ CHECK: andseq r2, r2, #6              @ encoding: [0x06,0x20,0x12,0x02]
@ CHECK: andseq r2, r2, r3              @ encoding: [0x03,0x20,0x12,0x00]
@ CHECK: andseq r2, r2, r3              @ encoding: [0x03,0x20,0x12,0x00]

orrseq r2, r2, #6
orrseq r2, #6
orrseq r2, r2, r3
orrseq r2, r3

@ CHECK: orrseq r2, r2, #6              @ encoding: [0x06,0x20,0x92,0x03]
@ CHECK: orrseq r2, r2, #6              @ encoding: [0x06,0x20,0x92,0x03]
@ CHECK: orrseq r2, r2, r3              @ encoding: [0x03,0x20,0x92,0x01]
@ CHECK: orrseq r2, r2, r3              @ encoding: [0x03,0x20,0x92,0x01]

eorseq r2, r2, #6
eorseq r2, #6
eorseq r2, r2, r3
eorseq r2, r3

@ CHECK: eorseq r2, r2, #6              @ encoding: [0x06,0x20,0x32,0x02]
@ CHECK: eorseq r2, r2, #6              @ encoding: [0x06,0x20,0x32,0x02]
@ CHECK: eorseq r2, r2, r3              @ encoding: [0x03,0x20,0x32,0x00]
@ CHECK: eorseq r2, r2, r3              @ encoding: [0x03,0x20,0x32,0x00]

bicseq r2, r2, #6
bicseq r2, #6
bicseq r2, r2, r3
bicseq r2, r3

@ CHECK: bicseq r2, r2, #6              @ encoding: [0x06,0x20,0xd2,0x03]
@ CHECK: bicseq r2, r2, #6              @ encoding: [0x06,0x20,0xd2,0x03]
@ CHECK: bicseq r2, r2, r3              @ encoding: [0x03,0x20,0xd2,0x01]
@ CHECK: bicseq r2, r2, r3              @ encoding: [0x03,0x20,0xd2,0x01]

add r0, pc, #123

@ CHECK: adr	r0, #123                @ encoding: [0x7b,0x00,0x8f,0xe2]
