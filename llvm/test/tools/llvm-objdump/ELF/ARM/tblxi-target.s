## Check that the disassembler reports the target address of a Thumb BLX(i)
## instruction correctly even if the instruction is not 32-bit aligned.

# RUN: llvm-mc %s --triple=armv8a -filetype=obj | \
# RUN:   llvm-objdump --no-print-imm-hex -dr - --triple armv8a --no-show-raw-insn | \
# RUN:   FileCheck %s

# CHECK:      00000000 <foo>:
# CHECK:      00000004 <test>:
# CHECK-NEXT:   4:  nop
# CHECK-NEXT:   6:  blx  0x0 <foo>   @ imm = #-8
# CHECK-NEXT:   a:  blx  0x10 <bar>  @ imm = #4
# CHECK:      00000010 <bar>:

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
