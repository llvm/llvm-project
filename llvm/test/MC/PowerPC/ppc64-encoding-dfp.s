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
# CHECK-LE: dctdp 8, 2                     # encoding: [0x04,0x12,0x00,0xed]
# CHECK-BE: dctdp 8, 2                     # encoding: [0xed,0x00,0x12,0x04]
            dctdp 8, 2
# CHECK-LE: dctdp. 8, 2                     # encoding: [0x05,0x12,0x00,0xed]
# CHECK-BE: dctdp. 8, 2                     # encoding: [0xed,0x00,0x12,0x05]
            dctdp. 8, 2
# CHECK-LE: dctqpq 8, 2                     # encoding: [0x04,0x12,0x00,0xfd]
# CHECK-BE: dctqpq 8, 2                     # encoding: [0xfd,0x00,0x12,0x04]
            dctqpq 8, 2
# CHECK-LE: dctqpq. 8, 2                    # encoding: [0x05,0x12,0x00,0xfd]
# CHECK-BE: dctqpq. 8, 2                    # encoding: [0xfd,0x00,0x12,0x05]
            dctqpq. 8, 2
# CHECK-LE: drsp 20, 8                      # encoding: [0x04,0x46,0x80,0xee]
# CHECK-BE: drsp 20, 8                      # encoding: [0xee,0x80,0x46,0x04]
            drsp 20, 8
# CHECK-LE: drsp. 20, 8                     # encoding: [0x05,0x46,0x80,0xee]
# CHECK-BE: drsp. 20, 8                     # encoding: [0xee,0x80,0x46,0x05]
            drsp. 20, 8
# CHECK-LE: drdpq 20, 8                     # encoding: [0x04,0x46,0x80,0xfe]
# CHECK-BE: drdpq 20, 8                     # encoding: [0xfe,0x80,0x46,0x04]
            drdpq 20, 8
# CHECK-LE: drdpq. 20, 8                    # encoding: [0x05,0x46,0x80,0xfe]
# CHECK-BE: drdpq. 20, 8                    # encoding: [0xfe,0x80,0x46,0x05]
            drdpq. 20, 8
# CHECK-LE: dcffix 12, 7                    # encoding: [0x44,0x3e,0x80,0xed]
# CHECK-BE: dcffix 12, 7                    # encoding: [0xed,0x80,0x3e,0x44]
            dcffix 12, 7
# CHECK-LE: dcffix. 12, 7                   # encoding: [0x45,0x3e,0x80,0xed]
# CHECK-BE: dcffix. 12, 7                   # encoding: [0xed,0x80,0x3e,0x45]
            dcffix. 12, 7
# CHECK-LE: dcffixq 12, 8                   # encoding: [0x44,0x46,0x80,0xfd]
# CHECK-BE: dcffixq 12, 8                   # encoding: [0xfd,0x80,0x46,0x44]
            dcffixq 12, 8
# CHECK-LE: dcffixq. 12, 8                  # encoding: [0x45,0x46,0x80,0xfd]
# CHECK-BE: dcffixq. 12, 8                  # encoding: [0xfd,0x80,0x46,0x45]
            dcffixq. 12, 8
# CHECK-LE: dctfix 8, 4                     # encoding: [0x44,0x22,0x00,0xed]
# CHECK-BE: dctfix 8, 4                     # encoding: [0xed,0x00,0x22,0x44]
            dctfix 8, 4
# CHECK-LE: dctfix. 8, 4                    # encoding: [0x45,0x22,0x00,0xed]
# CHECK-BE: dctfix. 8, 4                    # encoding: [0xed,0x00,0x22,0x45]
            dctfix. 8, 4
# CHECK-LE: dctfixq 8, 4                    # encoding: [0x44,0x22,0x00,0xfd]
# CHECK-BE: dctfixq 8, 4                    # encoding: [0xfd,0x00,0x22,0x44]
            dctfixq 8, 4
# CHECK-LE: dctfixq. 8, 4                   # encoding: [0x45,0x22,0x00,0xfd]
# CHECK-BE: dctfixq. 8, 4                   # encoding: [0xfd,0x00,0x22,0x45]
            dctfixq. 8, 4
# CHECK-LE: dcffixqq 18, 20                 # encoding: [0xc4,0xa7,0x40,0xfe]
# CHECK-BE: dcffixqq 18, 20                 # encoding: [0xfe,0x40,0xa7,0xc4]
            dcffixqq 18, 20
# CHECK-LE: dctfixqq 8, 10                  # encoding: [0xc4,0x57,0x01,0xfd]
# CHECK-BE: dctfixqq 8, 10                  # encoding: [0xfd,0x01,0x57,0xc4]
            dctfixqq 8, 10
# CHECK-BE: ddedpd 0, 8, 10                # encoding: [0xed,0x00,0x52,0x84]
# CHECK-LE: ddedpd 0, 8, 10                # encoding: [0x84,0x52,0x00,0xed]
            ddedpd 0, 8, 10
# CHECK-BE: ddedpd. 0, 8, 10               # encoding: [0xed,0x00,0x52,0x85]
# CHECK-LE: ddedpd. 0, 8, 10               # encoding: [0x85,0x52,0x00,0xed]
            ddedpd. 0, 8, 10
# CHECK-BE: ddedpdq 1, 8, 10               # encoding: [0xfd,0x08,0x52,0x84]
# CHECK-LE: ddedpdq 1, 8, 10               # encoding: [0x84,0x52,0x08,0xfd]
            ddedpdq 1, 8, 10
# CHECK-BE: ddedpdq. 1, 8, 10              # encoding: [0xfd,0x08,0x52,0x85]
# CHECK-LE: ddedpdq. 1, 8, 10              # encoding: [0x85,0x52,0x08,0xfd]
            ddedpdq. 1, 8, 10
# CHECK-BE: denbcd 1, 12, 16               # encoding: [0xed,0x90,0x86,0x84]
# CHECK-LE: denbcd 1, 12, 16               # encoding: [0x84,0x86,0x90,0xed]
            denbcd 1, 12, 16
# CHECK-BE: denbcd. 0, 12, 16              # encoding: [0xed,0x80,0x86,0x85]
# CHECK-LE: denbcd. 0, 12, 16              # encoding: [0x85,0x86,0x80,0xed]
            denbcd. 0, 12, 16
# CHECK-BE: denbcdq 1, 12, 16              # encoding: [0xfd,0x90,0x86,0x84]
# CHECK-LE: denbcdq 1, 12, 16              # encoding: [0x84,0x86,0x90,0xfd]
            denbcdq 1, 12, 16
# CHECK-BE: denbcdq. 0, 12, 16             # encoding: [0xfd,0x80,0x86,0x85]
# CHECK-LE: denbcdq. 0, 12, 16             # encoding: [0x85,0x86,0x80,0xfd]
            denbcdq. 0, 12, 16
# CHECK-BE: dxex 8, 20                     # encoding: [0xed,0x00,0xa2,0xc4]
# CHECK-LE: dxex 8, 20                     # encoding: [0xc4,0xa2,0x00,0xed]
            dxex 8, 20
# CHECK-BE: dxex. 8, 20                    # encoding: [0xed,0x00,0xa2,0xc5]
# CHECK-LE: dxex. 8, 20                    # encoding: [0xc5,0xa2,0x00,0xed]
            dxex. 8, 20
# CHECK-BE: dxexq 8, 20                    # encoding: [0xfd,0x00,0xa2,0xc4]
# CHECK-LE: dxexq 8, 20                    # encoding: [0xc4,0xa2,0x00,0xfd]
            dxexq 8, 20
# CHECK-BE: dxexq. 8, 20                   # encoding: [0xfd,0x00,0xa2,0xc5]
# CHECK-LE: dxexq. 8, 20                   # encoding: [0xc5,0xa2,0x00,0xfd]
            dxexq. 8, 20
# CHECK-BE: diex 8, 12, 18                 # encoding: [0xed,0x0c,0x96,0xc4]
# CHECK-LE: diex 8, 12, 18                 # encoding: [0xc4,0x96,0x0c,0xed]
            diex 8, 12, 18
# CHECK-BE: diex. 8, 12, 18                # encoding: [0xed,0x0c,0x96,0xc5]
# CHECK-LE: diex. 8, 12, 18                # encoding: [0xc5,0x96,0x0c,0xed]
            diex. 8, 12, 18
# CHECK-BE: diexq 8, 12, 18                # encoding: [0xfd,0x0c,0x96,0xc4]
# CHECK-LE: diexq 8, 12, 18                # encoding: [0xc4,0x96,0x0c,0xfd]
            diexq 8, 12, 18
# CHECK-BE: diexq. 8, 12, 18               # encoding: [0xfd,0x0c,0x96,0xc5]
# CHECK-LE: diexq. 8, 12, 18               # encoding: [0xc5,0x96,0x0c,0xfd]
            diexq. 8, 12, 18
# CHECK-BE: dscli 22, 4, 63                # encoding: [0xee,0xc4,0xfc,0x84]
# CHECK-LE: dscli 22, 4, 63                # encoding: [0x84,0xfc,0xc4,0xee]
            dscli 22, 4, 63
# CHECK-BE: dscli. 22, 4, 63               # encoding: [0xee,0xc4,0xfc,0x85]
# CHECK-LE: dscli. 22, 4, 63               # encoding: [0x85,0xfc,0xc4,0xee]
            dscli. 22, 4, 63
# CHECK-BE: dscliq 22, 4, 63               # encoding: [0xfe,0xc4,0xfc,0x84]
# CHECK-LE: dscliq 22, 4, 63               # encoding: [0x84,0xfc,0xc4,0xfe]
            dscliq 22, 4, 63
# CHECK-BE: dscliq. 22, 4, 63              # encoding: [0xfe,0xc4,0xfc,0x85]
# CHECK-LE: dscliq. 22, 4, 63              # encoding: [0x85,0xfc,0xc4,0xfe]
            dscliq. 22, 4, 63
# CHECK-BE: dscri 16, 10, 50               # encoding: [0xee,0x0a,0xc8,0xc4]
# CHECK-LE: dscri 16, 10, 50               # encoding: [0xc4,0xc8,0x0a,0xee]
            dscri 16, 10, 50
# CHECK-BE: dscri. 16, 10, 50              # encoding: [0xee,0x0a,0xc8,0xc5]
# CHECK-LE: dscri. 16, 10, 50              # encoding: [0xc5,0xc8,0x0a,0xee]
            dscri. 16, 10, 50
# CHECK-BE: dscriq 16, 10, 50              # encoding: [0xfe,0x0a,0xc8,0xc4]
# CHECK-LE: dscriq 16, 10, 50              # encoding: [0xc4,0xc8,0x0a,0xfe]
            dscriq 16, 10, 50
# CHECK-BE: dscriq. 16, 10, 50             # encoding: [0xfe,0x0a,0xc8,0xc5]
# CHECK-LE: dscriq. 16, 10, 50             # encoding: [0xc5,0xc8,0x0a,0xfe]
            dscriq. 16, 10, 50
# CHECK-BE: dtstdc 2, 6, 4                 # encoding: [0xed,0x06,0x11,0x84]
# CHECK-LE: dtstdc 2, 6, 4                 # encoding: [0x84,0x11,0x06,0xed]
            dtstdc 2, 6, 4
# CHECK-BE: dtstdcq 2, 6, 4                # encoding: [0xfd,0x06,0x11,0x84]
# CHECK-LE: dtstdcq 2, 6, 4                # encoding: [0x84,0x11,0x06,0xfd]
            dtstdcq 2, 6, 4
# CHECK-BE: dtstdg 2, 6, 4                 # encoding: [0xed,0x06,0x11,0xc4]
# CHECK-LE: dtstdg 2, 6, 4                 # encoding: [0xc4,0x11,0x06,0xed]
            dtstdg 2, 6, 4
# CHECK-BE: dtstdgq 2, 6, 4                # encoding: [0xfd,0x06,0x11,0xc4]
# CHECK-LE: dtstdgq 2, 6, 4                # encoding: [0xc4,0x11,0x06,0xfd]
            dtstdgq 2, 6, 4
# CHECK-BE: dtstex 2, 6, 4                 # encoding: [0xed,0x06,0x21,0x44]
# CHECK-LE: dtstex 2, 6, 4                 # encoding: [0x44,0x21,0x06,0xed]
            dtstex 2, 6, 4
# CHECK-BE: dtstexq 2, 6, 4                # encoding: [0xfd,0x06,0x21,0x44]
# CHECK-LE: dtstexq 2, 6, 4                # encoding: [0x44,0x21,0x06,0xfd]
            dtstexq 2, 6, 4
# CHECK-BE: dtstsf 2, 6, 4                 # encoding: [0xed,0x06,0x25,0x44]
# CHECK-LE: dtstsf 2, 6, 4                 # encoding: [0x44,0x25,0x06,0xed]
            dtstsf 2, 6, 4
# CHECK-BE: dtstsfq 2, 6, 4                # encoding: [0xfd,0x06,0x25,0x44]
# CHECK-LE: dtstsfq 2, 6, 4                # encoding: [0x44,0x25,0x06,0xfd]
            dtstsfq 2, 6, 4
# CHECK-BE: dtstsfi 2, 6, 4                # encoding: [0xed,0x06,0x25,0x46]
# CHECK-LE: dtstsfi 2, 6, 4                # encoding: [0x46,0x25,0x06,0xed]
            dtstsfi 2, 6, 4
# CHECK-BE: dtstsfiq 2, 6, 4               # encoding: [0xfd,0x06,0x25,0x46]
# CHECK-LE: dtstsfiq 2, 6, 4               # encoding: [0x46,0x25,0x06,0xfd]
            dtstsfiq 2, 6, 4
