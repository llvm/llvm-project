# RUN: llvm-mc %s -filetype=obj -triple=mipsel-unknown-linux \
# RUN: -mattr=micromips | llvm-readobj -r - \
# RUN: | FileCheck %s
# CHECK: Relocations [
# CHECK:     0x0 R_MIPS_32 bar 0x0
# CHECK:     0x4 R_MIPS_32 L1 0x0

  .set    micromips
  .type   bar,@function
bar:
L1:
  nop
  .data
  .4byte bar 
  .4byte L1 

