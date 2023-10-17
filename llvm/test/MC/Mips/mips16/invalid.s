# RUN: not llvm-mc -triple=mips -mcpu=mips32r2 -mattr=+mips16 < %s 2> %t
# RUN: FileCheck %s < %t

# Instructions which are invalid.

$label:
  nop 4         # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  nop $4        # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  nop $label    # CHECK: :[[@LINE]]:7: error: invalid operand for instruction

