# Instructions that are correctly rejected but emit a wrong or misleading error.
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 -mattr=micromips -mattr=+dsp 2>%t1
# RUN: FileCheck %s < %t1

  .set noat
  wrdsp $5, 128            # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  wrdsp $5, -1             # CHECK: :[[@LINE]]:13: error: expected 10-bit unsigned immediate
