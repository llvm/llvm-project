# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s


# CHECK-BE: dadd 2, 3, 4                   # encoding: [0xec,0x43,0x20,0x04]
# CHECK-LE: dadd 2, 3, 4                   # encoding: [0x04,0x20,0x43,0xec]
            dadd 2, 3, 4
# CHECK-BE: dadd. 2, 3, 4                  # encoding: [0xec,0x43,0x20,0x05]
# CHECK-LE: dadd. 2, 3, 4                  # encoding: [0x05,0x20,0x43,0xec]
            dadd. 2, 3, 4
# CHECK-BE: daddq 2, 6, 4                  # encoding: [0xfc,0x46,0x20,0x04]
# CHECK-LE: daddq 2, 6, 4                  # encoding: [0x04,0x20,0x46,0xfc]
            daddq 2, 6, 4
# CHECK-BE: daddq. 2, 6, 4                 # encoding: [0xfc,0x46,0x20,0x05]
# CHECK-LE: daddq. 2, 6, 4                 # encoding: [0x05,0x20,0x46,0xfc]
            daddq. 2, 6, 4
# CHECK-BE: dsub 2, 3, 4                   # encoding: [0xec,0x43,0x24,0x04]
# CHECK-LE: dsub 2, 3, 4                   # encoding: [0x04,0x24,0x43,0xec]
            dsub 2, 3, 4
# CHECK-BE: dsub. 2, 3, 4                  # encoding: [0xec,0x43,0x24,0x05]
# CHECK-LE: dsub. 2, 3, 4                  # encoding: [0x05,0x24,0x43,0xec]
            dsub. 2, 3, 4
# CHECK-BE: dsubq 2, 6, 4                  # encoding: [0xfc,0x46,0x24,0x04]
# CHECK-LE: dsubq 2, 6, 4                  # encoding: [0x04,0x24,0x46,0xfc]
            dsubq 2, 6, 4
# CHECK-BE: dsubq. 2, 6, 4                 # encoding: [0xfc,0x46,0x24,0x05]
# CHECK-LE: dsubq. 2, 6, 4                 # encoding: [0x05,0x24,0x46,0xfc]
            dsubq. 2, 6, 4
# CHECK-BE: dmul 2, 3, 4                   # encoding: [0xec,0x43,0x20,0x44]
# CHECK-LE: dmul 2, 3, 4                   # encoding: [0x44,0x20,0x43,0xec]
            dmul 2, 3, 4
# CHECK-BE: dmul. 2, 3, 4                  # encoding: [0xec,0x43,0x20,0x45]
# CHECK-LE: dmul. 2, 3, 4                  # encoding: [0x45,0x20,0x43,0xec]
            dmul. 2, 3, 4
# CHECK-BE: dmulq 2, 6, 4                  # encoding: [0xfc,0x46,0x20,0x44]
# CHECK-LE: dmulq 2, 6, 4                  # encoding: [0x44,0x20,0x46,0xfc]
            dmulq 2, 6, 4
# CHECK-BE: dmulq. 2, 6, 4                 # encoding: [0xfc,0x46,0x20,0x45]
# CHECK-LE: dmulq. 2, 6, 4                 # encoding: [0x45,0x20,0x46,0xfc]
            dmulq. 2, 6, 4
# CHECK-BE: ddiv 2, 3, 4                   # encoding: [0xec,0x43,0x24,0x44]
# CHECK-LE: ddiv 2, 3, 4                   # encoding: [0x44,0x24,0x43,0xec]
            ddiv 2, 3, 4
# CHECK-BE: ddiv. 2, 3, 4                  # encoding: [0xec,0x43,0x24,0x45]
# CHECK-LE: ddiv. 2, 3, 4                  # encoding: [0x45,0x24,0x43,0xec]
            ddiv. 2, 3, 4
# CHECK-BE: ddivq 2, 6, 4                  # encoding: [0xfc,0x46,0x24,0x44]
# CHECK-LE: ddivq 2, 6, 4                  # encoding: [0x44,0x24,0x46,0xfc]
            ddivq 2, 6, 4
# CHECK-BE: ddivq. 2, 6, 4                 # encoding: [0xfc,0x46,0x24,0x45]
# CHECK-LE: ddivq. 2, 6, 4                 # encoding: [0x45,0x24,0x46,0xfc]
            ddivq. 2, 6, 4
# CHECK-BE: dcmpu 2, 6, 4                  # encoding: [0xed,0x06,0x25,0x04]
# CHECK-LE: dcmpu 2, 6, 4                  # encoding: [0x04,0x25,0x06,0xed]
            dcmpu 2, 6, 4
# CHECK-BE: dcmpuq 2, 6, 4                 # encoding: [0xfd,0x06,0x25,0x04]
# CHECK-LE: dcmpuq 2, 6, 4                 # encoding: [0x04,0x25,0x06,0xfd]
            dcmpuq 2, 6, 4
# CHECK-BE: dcmpo 2, 6, 4                  # encoding: [0xed,0x06,0x21,0x04]
# CHECK-LE: dcmpo 2, 6, 4                  # encoding: [0x04,0x21,0x06,0xed]
            dcmpo 2, 6, 4
# CHECK-BE: dcmpoq 2, 6, 4                 # encoding: [0xfd,0x06,0x21,0x04]
# CHECK-LE: dcmpoq 2, 6, 4                 # encoding: [0x04,0x21,0x06,0xfd]
            dcmpoq 2, 6, 4
# CHECK-BE: dquai 15, 8, 4, 3              # encoding: [0xed,0x0f,0x26,0x86]
# CHECK-LE: dquai 15, 8, 4, 3              # encoding: [0x86,0x26,0x0f,0xed]
            dquai 15, 8, 4, 3
# CHECK-BE: dquai. 15, 8, 4, 3             # encoding: [0xed,0x0f,0x26,0x87]
# CHECK-LE: dquai. 15, 8, 4, 3             # encoding: [0x87,0x26,0x0f,0xed]
            dquai. 15, 8, 4, 3
# CHECK-BE: dquaiq 15, 8, 4, 3             # encoding: [0xfd,0x0f,0x26,0x86]
# CHECK-LE: dquaiq 15, 8, 4, 3             # encoding: [0x86,0x26,0x0f,0xfd]
            dquaiq 15, 8, 4, 3
# CHECK-BE: dquaiq. 15, 8, 4, 3            # encoding: [0xfd,0x0f,0x26,0x87]
# CHECK-LE: dquaiq. 15, 8, 4, 3            # encoding: [0x87,0x26,0x0f,0xfd]
            dquaiq. 15, 8, 4, 3
# CHECK-BE: dqua 7, 15, 4, 2               # encoding: [0xec,0xef,0x24,0x06]
# CHECK-LE: dqua 7, 15, 4, 2               # encoding: [0x06,0x24,0xef,0xec]
            dqua 7, 15, 4, 2
# CHECK-BE: dqua. 7, 15, 4, 2              # encoding: [0xec,0xef,0x24,0x07]
# CHECK-LE: dqua. 7, 15, 4, 2              # encoding: [0x07,0x24,0xef,0xec]
            dqua. 7, 15, 4, 2
# CHECK-BE: dquaq 6, 14, 4, 2              # encoding: [0xfc,0xce,0x24,0x06]
# CHECK-LE: dquaq 6, 14, 4, 2              # encoding: [0x06,0x24,0xce,0xfc]
            dquaq 6, 14, 4, 2
# CHECK-BE: dquaq. 6, 14, 4, 2             # encoding: [0xfc,0xce,0x24,0x07]
# CHECK-LE: dquaq. 6, 14, 4, 2             # encoding: [0x07,0x24,0xce,0xfc]
            dquaq. 6, 14, 4, 2
# CHECK-BE: drrnd 8, 12, 6, 2               # encoding: [0xed,0x0c,0x34,0x46]
# CHECK-LE: drrnd 8, 12, 6, 2               # encoding: [0x46,0x34,0x0c,0xed]
            drrnd 8, 12, 6, 2
# CHECK-BE: drrnd. 8, 12, 6, 2              # encoding: [0xed,0x0c,0x34,0x47]
# CHECK-LE: drrnd. 8, 12, 6, 2              # encoding: [0x47,0x34,0x0c,0xed]
            drrnd. 8, 12, 6, 2
# CHECK-BE: drrndq 8, 12, 6, 2              # encoding: [0xfd,0x0c,0x34,0x46]
# CHECK-LE: drrndq 8, 12, 6, 2              # encoding: [0x46,0x34,0x0c,0xfd]
            drrndq 8, 12, 6, 2
# CHECK-BE: drrndq. 8, 12, 6, 2             # encoding: [0xfd,0x0c,0x34,0x47]
# CHECK-LE: drrndq. 8, 12, 6, 2             # encoding: [0x47,0x34,0x0c,0xfd]
            drrndq. 8, 12, 6, 2
# CHECK-LE: drintx 0, 8, 10, 3             # encoding: [0xc6,0x56,0x00,0xed]
# CHECK-BE: drintx 0, 8, 10, 3             # encoding: [0xed,0x00,0x56,0xc6]
            drintx 0, 8, 10, 3
# CHECK-LE: drintx. 1, 8, 10, 3            # encoding: [0xc7,0x56,0x01,0xed]
# CHECK-BE: drintx. 1, 8, 10, 3            # encoding: [0xed,0x01,0x56,0xc7]
            drintx. 1, 8, 10, 3
# CHECK-LE: drintxq 1, 8, 10, 3            # encoding: [0xc6,0x56,0x01,0xfd]
# CHECK-BE: drintxq 1, 8, 10, 3            # encoding: [0xfd,0x01,0x56,0xc6]
            drintxq 1, 8, 10, 3
# CHECK-LE: drintxq. 0, 8, 10, 3           # encoding: [0xc7,0x56,0x00,0xfd]
# CHECK-BE: drintxq. 0, 8, 10, 3           # encoding: [0xfd,0x00,0x56,0xc7]
            drintxq. 0, 8, 10, 3
# CHECK-LE: drintn 1, 10, 6, 2             # encoding: [0xc6,0x35,0x41,0xed]
# CHECK-BE: drintn 1, 10, 6, 2             # encoding: [0xed,0x41,0x35,0xc6]
            drintn 1, 10, 6, 2
# CHECK-LE: drintn. 0, 10, 6, 2            # encoding: [0xc7,0x35,0x40,0xed]
# CHECK-BE: drintn. 0, 10, 6, 2            # encoding: [0xed,0x40,0x35,0xc7]
            drintn. 0, 10, 6, 2
# CHECK-LE: drintnq 0, 10, 6, 2            # encoding: [0xc6,0x35,0x40,0xfd]
# CHECK-BE: drintnq 0, 10, 6, 2            # encoding: [0xfd,0x40,0x35,0xc6]
            drintnq 0, 10, 6, 2
# CHECK-LE: drintnq. 1, 10, 6, 2           # encoding: [0xc7,0x35,0x41,0xfd]
# CHECK-BE: drintnq. 1, 10, 6, 2           # encoding: [0xfd,0x41,0x35,0xc7]
            drintnq. 1, 10, 6, 2
