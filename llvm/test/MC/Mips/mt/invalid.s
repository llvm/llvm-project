# RUN: not llvm-mc -arch=mips -mcpu=mips32 -mattr=+mt < %s 2>&1 | FileCheck %s
  dmt 4           # CHECK: error: invalid operand for instruction
  dmt $4, $5      # CHECK: error: invalid operand for instruction
  dmt $5, 0($4)   # CHECK: error: invalid operand for instruction
  emt 4           # CHECK: error: invalid operand for instruction
  emt $4, $5      # CHECK: error: invalid operand for instruction
  emt $5, 0($5)   # CHECK: error: invalid operand for instruction
