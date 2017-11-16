# RUN: not llvm-mc -arch=mips -mcpu=mips32r2 -mattr=+mt -show-encoding < %s 2>%t1
# RUN: FileCheck %s < %t1

  mftc0 $4, 0($5)     # CHECK: error: invalid operand for instruction
  mftc0 $4, 0($5), 1  # CHECK: error: invalid operand for instruction
  mftc0 $4, $5, -1    # CHECK: error: expected 3-bit unsigned immediate
  mftc0 $4, $5, 9     # CHECK: error: expected 3-bit unsigned immediate
  mftc0 $4, $5, $6    # CHECK: error: expected 3-bit unsigned immediate
  mftgpr $4, 0($5)    # CHECK: error: invalid operand for instruction
  mftgpr $4, $5, $6   # CHECK: error: invalid operand for instruction
  mftlo $3, 0($ac1)   # CHECK: error: invalid operand for instruction
  mftlo $4, $ac1, $4  # CHECK: error: invalid operand for instruction
  mfthi $3, 0($ac1)   # CHECK: error: invalid operand for instruction
  mfthi $4, $ac1, $4  # CHECK: error: invalid operand for instruction
  mftacx $3, 0($ac1)  # CHECK: error: invalid operand for instruction
  mftacx $4, $ac1, $4 # CHECK: error: invalid operand for instruction
  mftdsp $4, $5       # CHECK: error: invalid operand for instruction
  mftdsp $4, $f5      # CHECK: error: invalid operand for instruction
  mftdsp $4, $ac0     # CHECK: error: invalid operand for instruction
  mftc1 $4, 0($f4)    # CHECK: error: invalid operand for instruction
  mfthc1 $4, 0($f4)   # CHECK: error: invalid operand for instruction
  cftc1 $4, 0($f4)    # CHECK: error: invalid operand for instruction
  cftc1 $4, $f4, $5   # CHECK: error: invalid operand for instruction
