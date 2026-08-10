@ RUN: not llvm-mc -triple=armv7-apple-darwin -mcpu=cortex-a8 -show-encoding -mattr=-trustzone < %s | FileCheck %s -check-prefix=NOTZ
@ RUN: llvm-mc -triple=armv7-apple-darwin -mcpu=cortex-a8 -show-encoding -mattr=trustzone < %s | FileCheck %s -check-prefix=TZ
@ RUN: llvm-mc -triple=armv6kz -mcpu=arm1176jz-s -show-encoding < %s | FileCheck %s -check-prefix=TZ

  .syntax unified
  .globl _func

@ Check that the assembler processes SMC instructions when TrustZone support is 
@ active and that it rejects them when this feature is not enabled

_func:
@ CHECK: _func


@------------------------------------------------------------------------------
@ SMC
@------------------------------------------------------------------------------
        smi #0xf                        @ SMI is old (ARMv6KZ) name for SMC
        smceq #0

@ NOTZ-NOT: smc 	#15
@ NOTZ-NOT: smceq	#0
@ TZ: smc	#15                     @ encoding: [0x7f,0x00,0x60,0xe1]
@ TZ: smceq	#0                      @ encoding: [0x70,0x00,0x60,0x01]

