# RUN: llvm-mc -triple=hexagon -filetype=asm %s | FileCheck %s

# Make sure the assembler can parse and print the "s" flag for Hexaon's
# small-data section.
# CHECK: .section .sdata,"aws",@progbits

  .section ".sdata", "aws", @progbits
var:
  .word 0

