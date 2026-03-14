@ RUN: llvm-mc -triple=thumbv6m -show-encoding < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-V6M %s
@ RUN: llvm-mc -triple=thumbv7m -show-encoding < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-V7M %s

  .syntax unified

@ Check that the assembler can handle the documented syntax from the ARM ARM.
@ These tests test instruction encodings specific to v6m & v7m (FeatureMClass).

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
@ CHECK: mrs	r0, control             @ encoding: [0xef,0xf3,0x14,0x80]

@------------------------------------------------------------------------------
@ MSR
@------------------------------------------------------------------------------

        msr  apsr, r0
        msr  apsr_nzcvq, r0
        msr  iapsr, r0
        msr  iapsr_nzcvq, r0
        msr  eapsr, r0
        msr  eapsr_nzcvq, r0
        msr  xpsr, r0
        msr  xpsr_nzcvq, r0
        msr  ipsr, r0
        msr  epsr, r0
        msr  iepsr, r0
        msr  msp, r0
        msr  psp, r0
        msr  primask, r0
        msr  control, r0

@ CHECK-V6M: msr	apsr, r0                @ encoding: [0x80,0xf3,0x00,0x88]
@ CHECK-V6M: msr	apsr, r0                @ encoding: [0x80,0xf3,0x00,0x88]
@ CHECK-V6M: msr	iapsr, r0               @ encoding: [0x80,0xf3,0x01,0x88]
@ CHECK-V6M: msr	iapsr, r0               @ encoding: [0x80,0xf3,0x01,0x88]
@ CHECK-V6M: msr	eapsr, r0               @ encoding: [0x80,0xf3,0x02,0x88]
@ CHECK-V6M: msr	eapsr, r0               @ encoding: [0x80,0xf3,0x02,0x88]
@ CHECK-V6M: msr	xpsr, r0                @ encoding: [0x80,0xf3,0x03,0x88]
@ CHECK-V6M: msr	xpsr, r0                @ encoding: [0x80,0xf3,0x03,0x88]
@ CHECK-V7M: msr	apsr_nzcvq, r0          @ encoding: [0x80,0xf3,0x00,0x88]
@ CHECK-V7M: msr	apsr_nzcvq, r0          @ encoding: [0x80,0xf3,0x00,0x88]
@ CHECK-V7M: msr	iapsr_nzcvq, r0         @ encoding: [0x80,0xf3,0x01,0x88]
@ CHECK-V7M: msr	iapsr_nzcvq, r0         @ encoding: [0x80,0xf3,0x01,0x88]
@ CHECK-V7M: msr	eapsr_nzcvq, r0         @ encoding: [0x80,0xf3,0x02,0x88]
@ CHECK-V7M: msr	eapsr_nzcvq, r0         @ encoding: [0x80,0xf3,0x02,0x88]
@ CHECK-V7M: msr	xpsr_nzcvq, r0          @ encoding: [0x80,0xf3,0x03,0x88]
@ CHECK-V7M: msr	xpsr_nzcvq, r0          @ encoding: [0x80,0xf3,0x03,0x88]
@ CHECK: msr	ipsr, r0                @ encoding: [0x80,0xf3,0x05,0x88]
@ CHECK: msr	epsr, r0                @ encoding: [0x80,0xf3,0x06,0x88]
@ CHECK: msr	iepsr, r0               @ encoding: [0x80,0xf3,0x07,0x88]
@ CHECK: msr	msp, r0                 @ encoding: [0x80,0xf3,0x08,0x88]
@ CHECK: msr	psp, r0                 @ encoding: [0x80,0xf3,0x09,0x88]
@ CHECK: msr	primask, r0             @ encoding: [0x80,0xf3,0x10,0x88]
@ CHECK: msr	control, r0             @ encoding: [0x80,0xf3,0x14,0x88]
