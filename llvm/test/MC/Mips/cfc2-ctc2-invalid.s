# RUN: not llvm-mc %s -triple=mips-unknown-linux-gnu -mcpu=mips1 2>&1 \
# RUN:   | FileCheck %s

  cfc2 $2          # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  cfc2 $2, $32     # CHECK: :[[@LINE]]:12: error: invalid register number
  cfc2 $2, 5       # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  ctc2 $f1, $1     # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  cfc2 $2, $1, $3  # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
