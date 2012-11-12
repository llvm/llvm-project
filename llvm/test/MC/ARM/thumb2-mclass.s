@ RUN: llvm-mc -triple=thumbv7m-apple-darwin -show-encoding < %s | FileCheck %s
  .syntax unified
  .globl _func

@ Check that the assembler can handle the documented syntax from the ARM ARM.
@ These tests test instruction encodings specific to v7m & v7m (FeatureMClass).

@------------------------------------------------------------------------------
@ MRS
@------------------------------------------------------------------------------

        mrs  r0, apsr
        mrs  r0, iapsr
        mrs  r0, eapsr
        mrs  r0, xpsr
        mrs  r0, ipsr
        mrs  r0, epsr
        mrs  r0, iepsr
        mrs  r0, msp
        mrs  r0, psp
        mrs  r0, primask
        mrs  r0, basepri
        mrs  r0, basepri_max
        mrs  r0, faultmask
        mrs  r0, control

@ CHECK: mrs	r0, apsr                @ encoding: [0xef,0xf3,0x00,0x80]
@ CHECK: mrs	r0, iapsr               @ encoding: [0xef,0xf3,0x01,0x80]
@ CHECK: mrs	r0, eapsr               @ encoding: [0xef,0xf3,0x02,0x80]
@ CHECK: mrs	r0, xpsr                @ encoding: [0xef,0xf3,0x03,0x80]
@ CHECK: mrs	r0, ipsr                @ encoding: [0xef,0xf3,0x05,0x80]
@ CHECK: mrs	r0, epsr                @ encoding: [0xef,0xf3,0x06,0x80]
@ CHECK: mrs	r0, iepsr               @ encoding: [0xef,0xf3,0x07,0x80]
@ CHECK: mrs	r0, msp                 @ encoding: [0xef,0xf3,0x08,0x80]
@ CHECK: mrs	r0, psp                 @ encoding: [0xef,0xf3,0x09,0x80]
@ CHECK: mrs	r0, primask             @ encoding: [0xef,0xf3,0x10,0x80]
@ CHECK: mrs	r0, basepri             @ encoding: [0xef,0xf3,0x11,0x80]
@ CHECK: mrs	r0, basepri_max         @ encoding: [0xef,0xf3,0x12,0x80]
@ CHECK: mrs	r0, faultmask           @ encoding: [0xef,0xf3,0x13,0x80]
@ CHECK: mrs	r0, control             @ encoding: [0xef,0xf3,0x14,0x80]

@------------------------------------------------------------------------------
@ MSR
@------------------------------------------------------------------------------

        msr  apsr, r0
        msr  apsr_nzcvq, r0
        msr  apsr_g, r0
        msr  apsr_nzcvqg, r0
        msr  iapsr, r0
        msr  iapsr_nzcvq, r0
        msr  iapsr_g, r0
        msr  iapsr_nzcvqg, r0
        msr  eapsr, r0
        msr  eapsr_nzcvq, r0
        msr  eapsr_g, r0
        msr  eapsr_nzcvqg, r0
        msr  xpsr, r0
        msr  xpsr_nzcvq, r0
        msr  xpsr_g, r0
        msr  xpsr_nzcvqg, r0
        msr  ipsr, r0
        msr  epsr, r0
        msr  iepsr, r0
        msr  msp, r0
        msr  psp, r0
        msr  primask, r0
        msr  basepri, r0
        msr  basepri_max, r0
        msr  faultmask, r0
        msr  control, r0

@ CHECK: msr	apsr, r0                @ encoding: [0x80,0xf3,0x00,0x88]
@ CHECK: msr	apsr, r0                @ encoding: [0x80,0xf3,0x00,0x88]
@ CHECK: msr	apsr_g, r0              @ encoding: [0x80,0xf3,0x00,0x84]
@ CHECK: msr	apsr_nzcvqg, r0         @ encoding: [0x80,0xf3,0x00,0x8c]
@ CHECK: msr	iapsr, r0               @ encoding: [0x80,0xf3,0x01,0x88]
@ CHECK: msr	iapsr, r0               @ encoding: [0x80,0xf3,0x01,0x88]
@ CHECK: msr	iapsr_g, r0             @ encoding: [0x80,0xf3,0x01,0x84]
@ CHECK: msr	iapsr_nzcvqg, r0        @ encoding: [0x80,0xf3,0x01,0x8c]
@ CHECK: msr	eapsr, r0               @ encoding: [0x80,0xf3,0x02,0x88]
@ CHECK: msr	eapsr, r0               @ encoding: [0x80,0xf3,0x02,0x88]
@ CHECK: msr	eapsr_g, r0             @ encoding: [0x80,0xf3,0x02,0x84]
@ CHECK: msr	eapsr_nzcvqg, r0        @ encoding: [0x80,0xf3,0x02,0x8c]
@ CHECK: msr	xpsr, r0                @ encoding: [0x80,0xf3,0x03,0x88]
@ CHECK: msr	xpsr, r0                @ encoding: [0x80,0xf3,0x03,0x88]
@ CHECK: msr	xpsr_g, r0              @ encoding: [0x80,0xf3,0x03,0x84]
@ CHECK: msr	xpsr_nzcvqg, r0         @ encoding: [0x80,0xf3,0x03,0x8c]
@ CHECK: msr	ipsr, r0                @ encoding: [0x80,0xf3,0x05,0x88]
@ CHECK: msr	epsr, r0                @ encoding: [0x80,0xf3,0x06,0x88]
@ CHECK: msr	iepsr, r0               @ encoding: [0x80,0xf3,0x07,0x88]
@ CHECK: msr	msp, r0                 @ encoding: [0x80,0xf3,0x08,0x88]
@ CHECK: msr	psp, r0                 @ encoding: [0x80,0xf3,0x09,0x88]
@ CHECK: msr	primask, r0             @ encoding: [0x80,0xf3,0x10,0x88]
@ CHECK: msr	basepri, r0             @ encoding: [0x80,0xf3,0x11,0x88]
@ CHECK: msr	basepri_max, r0         @ encoding: [0x80,0xf3,0x12,0x88]
@ CHECK: msr	faultmask, r0           @ encoding: [0x80,0xf3,0x13,0x88]
@ CHECK: msr	control, r0             @ encoding: [0x80,0xf3,0x14,0x88]
