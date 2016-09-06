# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=micromips -mattr=+dspr3 | FileCheck %s

  .set noat
  bposge32c 342                # CHECK: bposge32c 342           # encoding: [0x43,0x20,0x00,0xab]
