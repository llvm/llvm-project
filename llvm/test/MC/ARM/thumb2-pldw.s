@ RUN: llvm-mc -triple=thumbv7-apple-darwin -mcpu=cortex-a8 -mattr=+mp -show-encoding < %s | FileCheck %s

@------------------------------------------------------------------------------
@ PLD(literal)
@------------------------------------------------------------------------------
         pldw   [r0, #257]
@ CHECK: pldw   [r0, #257]              @ encoding: [0xb0,0xf8,0x01,0xf1]
