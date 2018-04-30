# Instructions that are invalid.
#
# RUN: not llvm-mc %s -arch=mips -mcpu=mips64r5 -mattr=+virt 2>%t1
# RUN: FileCheck %s < %t1

  dmfgc0                # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  dmfgc0 0              # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  dmfgc0 $4             # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  dmfgc0 0, $4          # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  dmfgc0 0, $4, $5      # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  dmfgc0 $4, 0, $5      # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  dmfgc0 $4, $5, 8      # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  dmfgc0 $4, $5, -1     # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  dmfgc0 $4, $5, 0($4)  # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
  dmtgc0                # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  dmtgc0 0              # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  dmtgc0 $4             # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  dmtgc0 0, $4          # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  dmtgc0 0, $4, $5      # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  dmtgc0 $4, 0, $5      # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  dmtgc0 $4, $5, 8      # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  dmtgc0 $4, $5, -1     # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  dmtgc0 $4, $5, 0($4)  # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
