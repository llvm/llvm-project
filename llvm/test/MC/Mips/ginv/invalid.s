# Instructions that are invalid.
#
# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r6 -mattr=+ginv 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64r6 -mattr=+ginv 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r6 \
# RUN:     -mattr=+micromips,+ginv 2>%t1
# RUN: FileCheck %s < %t1

  ginvi            # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  ginvi 0          # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  ginvi $4, 0      # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
  ginvi $4, $5     # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
  ginvi 0($4)      # CHECK: :[[@LINE]]:10: error: unexpected token in argument list
  ginvt            # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  ginvt 0          # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  ginvt $4         # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  ginvt $4, $5     # CHECK: :[[@LINE]]:13: error: expected 2-bit unsigned immediate
  ginvt $4, 4      # CHECK: :[[@LINE]]:13: error: expected 2-bit unsigned immediate
  ginvt $4, -1     # CHECK: :[[@LINE]]:13: error: expected 2-bit unsigned immediate
  ginvt $4, 0, 1   # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
  ginvt $4, 0($4)  # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
