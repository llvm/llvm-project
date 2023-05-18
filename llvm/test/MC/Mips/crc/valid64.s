# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips64r6 -mattr=+crc | FileCheck %s

  .set noat
  crc32d $10, $11, $10   # CHECK: crc32d  $10, $11, $10  # encoding: [0x7d,0x6a,0x00,0xcf]
  crc32cd $10, $11, $10  # CHECK: crc32cd $10, $11, $10  # encoding: [0x7d,0x6a,0x01,0xcf]
