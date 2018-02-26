# RUN: not llvm-mc -arch=mips -mcpu=mips32r2 -mattr=+mt -show-encoding < %s 2>%t1
# RUN: FileCheck %s < %t1

# The integrated assembler produces a wrong or misleading error message.

  mftc0 0($4), $5    # CHECK: error: unexpected token in argument list
  mftc0 0($4), $5, 1 # CHECK: error: unexpected token in argument list
  mftgpr 0($4), $5   # CHECK: error: unexpected token in argument list
  mftlo 0($3)        # CHECK: error: unexpected token in argument list
  mftlo 0($3), $ac1  # CHECK: error: unexpected token in argument list
  mfthi 0($3)        # CHECK: error: unexpected token in argument list
  mfthi 0($3), $ac1  # CHECK: error: unexpected token in argument list
  mftacx 0($3)       # CHECK: error: unexpected token in argument list
  mftacx 0($3), $ac1 # CHECK: error: unexpected token in argument list
  mftdsp 0($4)       # CHECK: error: unexpected token in argument list
  mftc1 0($4), $f4   # CHECK: error: unexpected token in argument list
  mfthc1 0($4), $f4  # CHECK: error: unexpected token in argument list
  cftc1 0($4), $f8   # CHECK: error: unexpected token in argument list
