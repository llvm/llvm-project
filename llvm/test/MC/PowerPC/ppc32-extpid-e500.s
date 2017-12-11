# RUN: llvm-mc -triple powerpc-unknown-unknown --show-encoding %s | FileCheck %s

# Extended PID instructions specific to the e500 / e500mc cores:

# CHECK: lbepx    1, 2, 3                  # encoding: [0x7c,0x22,0x18,0xbe]
         lbepx    1, 2, 3
# CHECK: lfdepx   4, 5, 6                  # encoding: [0x7c,0x85,0x34,0xbe]
         lfdepx   4, 5, 6
# CHECK: lhepx    7, 8, 9                  # encoding: [0x7c,0xe8,0x4a,0x3e]
         lhepx    7, 8, 9
# CHECK: lwepx    10, 11, 12               # encoding: [0x7d,0x4b,0x60,0x3e]
         lwepx    10, 11, 12
# CHECK: stbepx   13, 14, 15               # encoding: [0x7d,0xae,0x79,0xbe]
         stbepx   13, 14, 15
# CHECK: stfdepx  16, 17, 18               # encoding: [0x7e,0x11,0x95,0xbe]
         stfdepx  16, 17, 18
# CHECK: sthepx   19, 20, 21               # encoding: [0x7e,0x74,0xab,0x3e]
         sthepx   19, 20, 21
# CHECK: stwepx   22, 23, 24               # encoding: [0x7e,0xd7,0xc1,0x3e]
         stwepx   22, 23, 24
# CHECK: dcbfep   25, 26                   # encoding: [0x7c,0x19,0xd0,0xfe]
         dcbfep   25, 26
# CHECK: dcbstep  27, 28                   # encoding: [0x7c,0x1b,0xe0,0x7e]
         dcbstep  27, 28
# CHECK: dcbtep   29, 30, 31               # encoding: [0x7f,0xbe,0xfa,0x7e]
         dcbtep   29, 30, 31
# CHECK: dcbtstep 0, 1, 2                  # encoding: [0x7c,0x01,0x11,0xfe]
         dcbtstep 0, 1, 2
# CHECK: dcbzep   3, 4                     # encoding: [0x7c,0x03,0x27,0xfe]
         dcbzep   3, 4
# CHECK: dcbzlep  5, 6                     # encoding: [0x7c,0x25,0x37,0xfe]
         dcbzlep  5, 6
# CHECK: icbiep   7, 8                     # encoding: [0x7c,0x07,0x47,0xbe]
         icbiep   7, 8
