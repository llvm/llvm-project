@ PR17647: MUL/MLA/SMLAL/UMLAL should be avalaibe to IAS for ARMv4 and higher

@ RUN: llvm-mc < %s -triple armv4-unknown-unknown -show-encoding | FileCheck %s --check-prefix=ARMV4

@ ARMV4: mul	r0, r1, r2              @ encoding: [0x91,0x02,0x00,0xe0]
@ ARMV4: muls	r0, r1, r2              @ encoding: [0x91,0x02,0x10,0xe0]
@ ARMV4: mulne	r0, r1, r2              @ encoding: [0x91,0x02,0x00,0x10]
@ ARMV4: mulseq	r0, r1, r2              @ encoding: [0x91,0x02,0x10,0x00]
mul r0, r1, r2
muls r0, r1, r2
mulne r0, r1, r2
mulseq r0, r1, r2

@ ARMV4: mla	r0, r1, r2, r3          @ encoding: [0x91,0x32,0x20,0xe0]
@ ARMV4: mlas	r0, r1, r2, r3          @ encoding: [0x91,0x32,0x30,0xe0]
@ ARMV4: mlane	r0, r1, r2, r3          @ encoding: [0x91,0x32,0x20,0x10]
@ ARMV4: mlaseq	r0, r1, r2, r3          @ encoding: [0x91,0x32,0x30,0x00]
mla r0, r1, r2, r3
mlas r0, r1, r2, r3
mlane r0, r1, r2, r3
mlaseq r0, r1, r2, r3

@ ARMV4: smlal	r2, r3, r0, r1          @ encoding: [0x90,0x21,0xe3,0xe0]
@ ARMV4: smlals	r2, r3, r0, r1          @ encoding: [0x90,0x21,0xf3,0xe0]
@ ARMV4: smlalne	r2, r3, r0, r1          @ encoding: [0x90,0x21,0xe3,0x10]
@ ARMV4: smlalseq	r2, r3, r0, r1  @ encoding: [0x90,0x21,0xf3,0x00]
smlal r2,r3,r0,r1
smlals r2,r3,r0,r1
smlalne r2,r3,r0,r1
smlalseq r2,r3,r0,r1

@ ARMV4: umlal	r2, r3, r0, r1          @ encoding: [0x90,0x21,0xa3,0xe0]
@ ARMV4: umlals	r2, r3, r0, r1          @ encoding: [0x90,0x21,0xb3,0xe0]
@ ARMV4: umlalne	r2, r3, r0, r1          @ encoding: [0x90,0x21,0xa3,0x10]
@ ARMV4: umlalseq	r2, r3, r0, r1  @ encoding: [0x90,0x21,0xb3,0x00]
umlal r2,r3,r0,r1
umlals r2,r3,r0,r1
umlalne r2,r3,r0,r1
umlalseq r2,r3,r0,r1
