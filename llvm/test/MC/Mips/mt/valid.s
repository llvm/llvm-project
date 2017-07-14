# RUN: llvm-mc -arch=mips -mcpu=mips32r2 -mattr=+mt -show-encoding < %s \
# RUN:   | FileCheck %s
  dmt           # CHECK:  dmt         # encoding: [0x41,0x60,0x0b,0xc1]
  dmt $5        # CHECK:  dmt $5      # encoding: [0x41,0x65,0x0b,0xc1]
  emt           # CHECK:  emt         # encoding: [0x41,0x60,0x0b,0xe1]
  emt $4        # CHECK:  emt $4      # encoding: [0x41,0x64,0x0b,0xe1]
  dvpe          # CHECK:  dvpe        # encoding: [0x41,0x60,0x00,0x01]
  dvpe $6       # CHECK:  dvpe  $6    # encoding: [0x41,0x66,0x00,0x01]
  evpe          # CHECK:  evpe        # encoding: [0x41,0x60,0x00,0x21]
  evpe $4       # CHECK:  evpe  $4    # encoding: [0x41,0x64,0x00,0x21]
  fork $2, $3, $5 # CHECK:  fork  $2, $3, $5 # encoding: [0x7c,0x65,0x10,0x08]
  yield $4        # CHECK:  yield  $4        # encoding: [0x7c,0x80,0x00,0x09]
  yield $4, $5    # CHECK:  yield $4, $5     # encoding: [0x7c,0xa0,0x20,0x09]
