## Check that the disassembler reports the target address of a Thumb BLX(i)
## instruction correctly even if the instruction is not 32-bit aligned.

# RUN: llvm-mc %s --triple=armv8a -filetype=obj | \
# RUN:   llvm-objdump -dr - --triple armv8a --no-show-raw-insn | \
# RUN:   FileCheck %s

# CHECK:      <test>:
# CHECK-NEXT:   4:  nop
# CHECK-NEXT:   6:  blx  #-8 <foo>
# CHECK-NEXT:   a:  blx  #4 <bar>

  .arm
foo:
  nop

  .thumb
test:
  nop
  blx #-8
  blx #4

  .arm
  .p2align 2
bar:
  nop
