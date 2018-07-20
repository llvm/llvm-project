# Instructions that are invalid.
#
# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r6 -mattr=+crc 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64r6 -mattr=+crc 2>%t1
# RUN: FileCheck %s < %t1

  .set noat
  crc32b  $1, $2, $2      # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32b  $1, $2, $3      # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32b  $1, $2, 2       # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
  crc32b  $1, 2, $2       # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
  crc32b  1, $2, $2       # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  crc32b  $1, $2          # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32b  $1              # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32b  $1, $2, 0($2)   # CHECK: :[[@LINE]]:19: error: invalid operand for instruction

  crc32h  $1, $2, $2      # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32h  $1, $2, $3      # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32h  $1, $2, 2       # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
  crc32h  $1, 2, $2       # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
  crc32h  1, $2, $2       # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  crc32h  $1, $2          # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32h  $1              # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32h  $1, $2, 0($2)   # CHECK: :[[@LINE]]:19: error: invalid operand for instruction

  crc32w  $1, $2, $2      # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32w  $1, $2, $3      # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32w  $1, $2, 2       # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
  crc32w  $1, 2, $2       # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
  crc32w  1, $2, $2       # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  crc32w  $1, $2          # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32w  $1              # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32w  $1, $2, 0($2)   # CHECK: :[[@LINE]]:19: error: invalid operand for instruction

  crc32cb  $1, $2, $2     # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32cb  $1, $2, $3     # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32cb  $1, $2, 2      # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
  crc32cb  $1, 2, $2      # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
  crc32cb  1, $2, $2      # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  crc32cb  $1, $2         # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32cb  $1             # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32cb  $1, $2, 0($2)  # CHECK: :[[@LINE]]:20: error: invalid operand for instruction

  crc32ch  $1, $2, $2     # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32ch  $1, $2, $3     # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32ch  $1, $2, 2      # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
  crc32ch  $1, 2, $2      # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
  crc32ch  1, $2, $2      # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  crc32ch  $1, $2         # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32ch  $1             # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32ch  $1, $2, 0($2)  # CHECK: :[[@LINE]]:20: error: invalid operand for instruction

  crc32cw  $1, $2, $2     # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32cw  $1, $2, $3     # CHECK: :[[@LINE]]:3: error: source and destination must match
  crc32cw  $1, $2, 2      # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
  crc32cw  $1, 2, $2      # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
  crc32cw  1, $2, $2      # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  crc32cw  $1, $2         # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32cw  $1             # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  crc32cw  $1, $2, 0($2)  # CHECK: :[[@LINE]]:20: error: invalid operand for instruction

  crc32 $1, $2, $2        # CHECK: :[[@LINE]]:3: error: unknown instruction
  crcb $1, $2, $2         # CHECK: :[[@LINE]]:3: error: unknown instruction
  crc $1, $2, $2          # CHECK: :[[@LINE]]:3: error: unknown instruction
