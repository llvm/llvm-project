@ RUN: not llvm-mc -triple=thumbv7m 2>&1 < %s | FileCheck --check-prefix=CHECK-ERRORS %s
@ RUN: llvm-mc -triple=thumbv7em -show-encoding < %s | FileCheck --check-prefix=CHECK-7EM %s

sxtab r0, r0, r0
sxtah r0, r0, r0
sxtab16 r0, r0, r0
sxtb16 r0, r0
sxtb16 r0, r0, ror #8
@ CHECK-ERRORS: error: instruction requires: dsp
@ CHECK-ERRORS: error: instruction requires: dsp
@ CHECK-ERRORS: error: instruction requires: dsp
@ CHECK-ERRORS: error: instruction requires: dsp
@ CHECK-ERRORS: error: invalid instruction
@ CHECK-7EM: sxtab	r0, r0, r0              @ encoding: [0x40,0xfa,0x80,0xf0]
@ CHECK-7EM: sxtah	r0, r0, r0              @ encoding: [0x00,0xfa,0x80,0xf0]
@ CHECK-7EM: sxtab16	r0, r0, r0              @ encoding: [0x20,0xfa,0x80,0xf0]
@ CHECK-7EM: sxtb16	r0, r0                  @ encoding: [0x2f,0xfa,0x80,0xf0]
@ CHECK-7EM: sxtb16	r0, r0, ror #8          @ encoding: [0x2f,0xfa,0x90,0xf0]

uxtab r0, r0, r0
uxtah r0, r0, r0
uxtab16 r0, r0, r0
uxtb16 r0, r0
uxtb16 r0, r0, ror #8
@ CHECK-ERRORS: error: instruction requires: dsp
@ CHECK-ERRORS: error: instruction requires: dsp
@ CHECK-ERRORS: error: instruction requires: dsp
@ CHECK-ERRORS: error: instruction requires: dsp
@ CHECK-ERRORS: error: invalid instruction
@ CHECK-7EM: uxtab	r0, r0, r0              @ encoding: [0x50,0xfa,0x80,0xf0]
@ CHECK-7EM: uxtah	r0, r0, r0              @ encoding: [0x10,0xfa,0x80,0xf0]
@ CHECK-7EM: uxtab16	r0, r0, r0              @ encoding: [0x30,0xfa,0x80,0xf0]
@ CHECK-7EM: uxtb16	r0, r0                  @ encoding: [0x3f,0xfa,0x80,0xf0]
@ CHECK-7EM: uxtb16	r0, r0, ror #8          @ encoding: [0x3f,0xfa,0x90,0xf0]
