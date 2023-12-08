# Instructions that are invalid.
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux-gnu -mcpu=mips64r6 \
# RUN:     -mattr=+crc 2>%t1
# RUN: FileCheck %s < %t1

  .set noat
  crc32d $1, $2, $2       # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32d $1, $2, $3       # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32d $1, $2, 2        # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
  crc32d $1, 2, $2        # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  crc32d 1, $2, $2        # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  crc32d $1, $2           # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32d $1               # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32d $1, $2, 0($2)    # CHECK: :[[@LINE]]:18: error: invalid operand for instruction

  crc32cd $1, $2, $2      # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32cd $1, $2, $3      # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32cd $1, $2, 2       # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
  crc32cd $1, 2, $2       # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
  crc32cd 1, $2, $2       # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  crc32cd $1, $2          # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32cd $1              # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32cd $1, $2, 0($2)   # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
