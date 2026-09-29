# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips32r6 -mattr=+crc | FileCheck %s
# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips64r6 -mattr=+crc | FileCheck %s

  .set noat
  crc32b $1, $2, $1      # CHECK: crc32b $1, $2, $1   # encoding: [0x7c,0x41,0x00,0x0f]
  crc32h $4, $5, $4      # CHECK: crc32h $4, $5, $4   # encoding: [0x7c,0xa4,0x00,0x4f]
  crc32w $7, $8, $7      # CHECK: crc32w $7, $8, $7   # encoding: [0x7d,0x07,0x00,0x8f]
  crc32cb $1, $2, $1     # CHECK: crc32cb $1, $2, $1  # encoding: [0x7c,0x41,0x01,0x0f]
  crc32ch $4, $5, $4     # CHECK: crc32ch $4, $5, $4  # encoding: [0x7c,0xa4,0x01,0x4f]
  crc32cw $7, $8, $7     # CHECK: crc32cw $7, $8, $7  # encoding: [0x7d,0x07,0x01,0x8f]
