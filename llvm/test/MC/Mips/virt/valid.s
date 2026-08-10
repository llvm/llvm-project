# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips32r5 -mattr=+virt | FileCheck %s
# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips64r5 -mattr=+virt | FileCheck %s

  mfgc0 $4, $5           # CHECK: mfgc0 $4, $5, 0   # encoding: [0x40,0x64,0x28,0x00]
  mfgc0 $4, $5, 2        # CHECK: mfgc0 $4, $5, 2   # encoding: [0x40,0x64,0x28,0x02]
  mtgc0 $4, $5           # CHECK: mtgc0 $4, $5, 0   # encoding: [0x40,0x64,0x2a,0x00]
  mtgc0 $5, $4, 2        # CHECK: mtgc0 $5, $4, 2   # encoding: [0x40,0x65,0x22,0x02]
  mthgc0 $5, $4          # CHECK: mthgc0 $5, $4, 0  # encoding: [0x40,0x65,0x26,0x00]
  mthgc0 $5, $4, 1       # CHECK: mthgc0 $5, $4, 1  # encoding: [0x40,0x65,0x26,0x01]
  mfhgc0 $5, $4          # CHECK: mfhgc0 $5, $4, 0  # encoding: [0x40,0x65,0x24,0x00]
  mfhgc0 $5, $4, 7       # CHECK: mfhgc0 $5, $4, 7  # encoding: [0x40,0x65,0x24,0x07]
  hypcall                # CHECK: hypcall           # encoding: [0x42,0x00,0x00,0x28]
  hypcall 10             # CHECK: hypcall 10        # encoding: [0x42,0x00,0x50,0x28]
  tlbginv                # CHECK: tlbginv           # encoding: [0x42,0x00,0x00,0x0b]
  tlbginvf               # CHECK: tlbginvf          # encoding: [0x42,0x00,0x00,0x0c]
  tlbgp                  # CHECK: tlbgp             # encoding: [0x42,0x00,0x00,0x10]
  tlbgr                  # CHECK: tlbgr             # encoding: [0x42,0x00,0x00,0x09]
  tlbgwi                 # CHECK: tlbgwi            # encoding: [0x42,0x00,0x00,0x0a]
  tlbgwr                 # CHECK: tlbgwr            # encoding: [0x42,0x00,0x00,0x0e]
