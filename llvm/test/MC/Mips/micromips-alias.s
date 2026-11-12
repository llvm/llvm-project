# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux -mcpu=mips32 %s -o - \
# RUN:   | llvm-readobj --symbols - | FileCheck %s

# Symbol bar must be marked as micromips.
# CHECK: Name: bar
# CHECK: Other [ (0x80)
  .align 2
  .type  f,@function
  .set   nomips16
  .set   micromips
f:
  nop
  .set   nomicromips
  nop
  .globl bar
bar = f

# CHECK: Name: foo
# CHECK: Other [ (0x80)
  .type  o,@object
  .set   micromips
o:
  .insn
  .word 0x00000000
  .set   nomicromips

  .globl foo
foo = o
