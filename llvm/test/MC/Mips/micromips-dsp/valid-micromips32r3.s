# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r3 -mattr=micromips -mattr=+dsp | FileCheck %s

  .set noat
  bposge32 342                 # CHECK: bposge32 342            # encoding: [0x43,0x60,0x00,0xab]
