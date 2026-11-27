# Instructions that are invalid.
#
# RUN: not llvm-mc %s -triple nanomips-elf \
# RUN:                -mattr=+ginv 2>%t1
# RUN: FileCheck %s < %t1

  ginvi              # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  ginvi 0            # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  ginvi $a0, 0       # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  ginvi $a0, $a1     # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  ginvi 0($a0)       # CHECK: :[[@LINE]]:10: error: unexpected token in argument list
  ginvt              # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  ginvt 0            # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  ginvt $a0          # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  ginvt $a0, $a1     # CHECK: :[[@LINE]]:14: error: expected 2-bit unsigned immediate
  ginvt $a0, 4       # CHECK: :[[@LINE]]:14: error: expected 2-bit unsigned immediate
  ginvt $a0, -1      # CHECK: :[[@LINE]]:14: error: expected 2-bit unsigned immediate
  ginvt $a0, 0, 1    # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
  ginvt $a0, 0($a0)  # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
