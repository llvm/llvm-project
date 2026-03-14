# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips32r5 -mattr=+micromips,+virt | FileCheck %s

  mfgc0 $4, $5        # CHECK: mfgc0 $4, $5, 0   # encoding: [0x00,0x85,0x04,0xfc]
  mfgc0 $4, $5, 2     # CHECK: mfgc0 $4, $5, 2   # encoding: [0x00,0x85,0x14,0xfc]
  mtgc0 $5, $4        # CHECK: mtgc0 $5, $4, 0   # encoding: [0x00,0xa4,0x06,0xfc]
  mtgc0 $5, $4, 2     # CHECK: mtgc0 $5, $4, 2   # encoding: [0x00,0xa4,0x16,0xfc]
  mthgc0 $5, $4       # CHECK: mthgc0 $5, $4, 0  # encoding: [0x00,0xa4,0x06,0xf4]
  mthgc0 $5, $4, 1    # CHECK: mthgc0 $5, $4, 1  # encoding: [0x00,0xa4,0x0e,0xf4]
  mfhgc0 $5, $4       # CHECK: mfhgc0 $5, $4, 0  # encoding: [0x00,0xa4,0x04,0xf4]
  mfhgc0 $5, $4, 7    # CHECK: mfhgc0 $5, $4, 7  # encoding: [0x00,0xa4,0x3c,0xf4]
  hypcall             # CHECK: hypcall           # encoding: [0x00,0x00,0xc3,0x7c]
  hypcall 10          # CHECK: hypcall 10        # encoding: [0x00,0x0a,0xc3,0x7c]
  tlbginv             # CHECK: tlbginv           # encoding: [0x00,0x00,0x41,0x7c]
  tlbginvf            # CHECK: tlbginvf          # encoding: [0x00,0x00,0x51,0x7c]
  tlbgp               # CHECK: tlbgp             # encoding: [0x00,0x00,0x01,0x7c]
  tlbgr               # CHECK: tlbgr             # encoding: [0x00,0x00,0x11,0x7c]
  tlbgwi              # CHECK: tlbgwi            # encoding: [0x00,0x00,0x21,0x7c]
  tlbgwr              # CHECK: tlbgwr            # encoding: [0x00,0x00,0x31,0x7c]
