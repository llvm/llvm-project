# Check that llvm-mc accepts arithmetic expression
# as an argument of the %got relocation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s \
# RUN:   | llvm-objdump -d -r - | FileCheck %s

  .text
foo:
  lw      $t0,%got($loc+0x10004)($gp)
# CHECK: 0:       8f 88 00 01     lw      $8, 1($gp)
# CHECK:                  00000000:  R_MIPS_GOT16 .data
  addi    $t0,$t0,%lo($loc+0x10004)
# CHECK: 4:       21 08 00 04     addi    $8, $8, 4
# CHECK:                  00000004:  R_MIPS_LO16  .data

  .data
$loc:
  .word 0
  .space 0x10000
  .word 0
