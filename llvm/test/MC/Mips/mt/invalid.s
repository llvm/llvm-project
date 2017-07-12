# RUN: not llvm-mc -arch=mips -mcpu=mips32 -mattr=+mt < %s 2>&1 | FileCheck %s
  dmt 4                   # CHECK: error: invalid operand for instruction
  dmt $4, $5              # CHECK: error: invalid operand for instruction
  dmt $5, 0($4)           # CHECK: error: invalid operand for instruction
  emt 4                   # CHECK: error: invalid operand for instruction
  emt $4, $5              # CHECK: error: invalid operand for instruction
  emt $5, 0($5)           # CHECK: error: invalid operand for instruction
  dvpe 4                  # CHECK: error: invalid operand for instruction
  dvpe $4, $5             # CHECK: error: invalid operand for instruction
  dvpe $5, 0($4)          # CHECK: error: invalid operand for instruction
  evpe 4                  # CHECK: error: invalid operand for instruction
  evpe $4, $5             # CHECK: error: invalid operand for instruction
  evpe $5, 0($5)          # CHECK: error: invalid operand for instruction
  mftr $4, 0($5), 0, 0, 0 # CHECK: error: invalid operand for instruction
  mftr $4, $5, 2, 0, 0    # CHECK: error: expected 1-bit unsigned immediate
  mftr $4, $5, -1, 0, 0   # CHECK: error: expected 1-bit unsigned immediate
  mftr $4, $5, 0, 8, 0    # CHECK: error: expected 3-bit unsigned immediate
  mftr $4, $5, 0, -1, 0   # CHECK: error: expected 3-bit unsigned immediate
  mftr $4, $4, 0, 0, 2    # CHECK: error: expected 1-bit unsigned immediate
  mftr $4, $5, 0, 0, -1   # CHECK: error: expected 1-bit unsigned immediate
  mttr $4, 0($5), 0, 0, 0 # CHECK: error: invalid operand for instruction
  mttr $4, $5, 2, 0, 0    # CHECK: error: expected 1-bit unsigned immediate
  mttr $4, $5, -1, 0, 0   # CHECK: error: expected 1-bit unsigned immediate
  mttr $4, $5, 0, 8, 0    # CHECK: error: expected 3-bit unsigned immediate
  mttr $4, $5, 0, -1, 0   # CHECK: error: expected 3-bit unsigned immediate
  mttr $4, $4, 0, 0, 2    # CHECK: error: expected 1-bit unsigned immediate
  mttr $4, $5, 0, 0, -1   # CHECK: error: expected 1-bit unsigned immediate
