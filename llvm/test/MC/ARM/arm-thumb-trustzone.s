@ RUN: not llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 -show-encoding -mattr=-trustzone < %s | FileCheck %s -check-prefix=NOTZ
@ RUN: llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 -show-encoding -mattr=trustzone < %s | FileCheck %s -check-prefix=TZ
@ RUN: not llvm-mc -triple=thumbv6kz -mcpu=arm1176jzf-s -show-encoding < %s | FileCheck %s -check-prefix=NOTZ

  .syntax unified
  .globl _func

@ Check that the assembler processes SMC instructions when TrustZone support is 
@ active and that it rejects them when this feature is not enabled

_func:
@ CHECK: _func


@------------------------------------------------------------------------------
@ SMC
@------------------------------------------------------------------------------
        smc #0xf
        it eq
        smceq #0

@ NOTZ-NOT: smc 	#15
@ NOTZ-NOT: smceq	#0
@ TZ: smc	#15                     @ encoding: [0xff,0xf7,0x00,0x80]
@ TZ: it	eq                      @ encoding: [0x08,0xbf]
@ TZ: smceq	#0                      @ encoding: [0xf0,0xf7,0x00,0x80]
