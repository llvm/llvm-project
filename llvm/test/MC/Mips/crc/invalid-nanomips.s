# Instructions that are invalid.
#
# RUN: not llvm-mc %s -triple=nanomips-elf -mattr=+crc 2>%t1
# RUN: FileCheck %s < %t1

  .set noat
  crc32b  $t4, 2        # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
  crc32b  2, $t4        # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  crc32b  $at           # CHECK: :[[@LINE]]:3: error: too few operands for instruction

  crc32h  $t4, 2        # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
  crc32h  2, $t4        # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  crc32h  $at           # CHECK: :[[@LINE]]:3: error: too few operands for instruction

  crc32w  $t4, 2        # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
  crc32w  2, $t4        # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  crc32w  $at           # CHECK: :[[@LINE]]:3: error: too few operands for instruction

  crc32cb  $t4, 2       # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
  crc32cb  2, $t4       # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  crc32cb  $at          # CHECK: :[[@LINE]]:3: error: too few operands for instruction

  crc32ch  $t4, 2       # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
  crc32ch  2, $t4       # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  crc32ch  $at          # CHECK: :[[@LINE]]:3: error: too few operands for instruction

  crc32cw  $t4, 2       # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
  crc32cw  2, $t4       # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  crc32cw  $at          # CHECK: :[[@LINE]]:3: error: too few operands for instruction

  crc32 $at, $t4        # CHECK: :[[@LINE]]:3: error: unknown instruction
  crcb  $at, $t4        # CHECK: :[[@LINE]]:3: error: unknown instruction
  crc   $at, $t4        # CHECK: :[[@LINE]]:3: error: unknown instruction
