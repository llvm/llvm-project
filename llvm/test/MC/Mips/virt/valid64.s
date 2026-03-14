# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips64r5 -mattr=+virt | FileCheck %s

  dmfgc0 $4,$5      # CHECK: dmfgc0 $4, $5, 0  # encoding: [0x40,0x64,0x29,0x00]
  dmfgc0 $4,$5,4    # CHECK: dmfgc0 $4, $5, 4  # encoding: [0x40,0x64,0x29,0x04]
  dmtgc0 $4,$5      # CHECK: dmtgc0 $4, $5, 0  # encoding: [0x40,0x64,0x2b,0x00]
  dmtgc0 $4,$5,4    # CHECK: dmtgc0 $4, $5, 4  # encoding: [0x40,0x64,0x2b,0x04]
