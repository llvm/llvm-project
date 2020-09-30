# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-LE %s

# CHECK-BE: plxv 63, 8589934591(0), 1       # encoding: [0x04,0x11,0xff,0xff
# CHECK-BE-SAME:                                         0xcf,0xe0,0xff,0xff]
# CHECK-LE: plxv 63, 8589934591(0), 1       # encoding: [0xff,0xff,0x11,0x04
# CHECK-LE-SAME:                                         0xff,0xff,0xe0,0xcf]
            plxv 63, 8589934591(0), 1
# CHECK-BE: plxv 33, -8589934592(31), 0     # encoding: [0x04,0x02,0x00,0x00
# CHECK-BE-SAME:                                         0xcc,0x3f,0x00,0x00]
# CHECK-LE: plxv 33, -8589934592(31), 0     # encoding: [0x00,0x00,0x02,0x04
# CHECK-LE-SAME:                                         0x00,0x00,0x3f,0xcc]
            plxv 33, -8589934592(31), 0
# CHECK-BE: pstxv 63, 8589934591(0), 1      # encoding: [0x04,0x11,0xff,0xff
# CHECK-BE-SAME:                                         0xdf,0xe0,0xff,0xff]
# CHECK-LE: pstxv 63, 8589934591(0), 1      # encoding: [0xff,0xff,0x11,0x04
# CHECK-LE-SAME:                                         0xff,0xff,0xe0,0xdf]
            pstxv 63, 8589934591(0), 1
# CHECK-BE: pstxv 33, -8589934592(31), 0    # encoding: [0x04,0x02,0x00,0x00
# CHECK-BE-SAME:                                         0xdc,0x3f,0x00,0x00]
# CHECK-LE: pstxv 33, -8589934592(31), 0    # encoding: [0x00,0x00,0x02,0x04
# CHECK-LE-SAME:                                         0x00,0x00,0x3f,0xdc]
            pstxv 33, -8589934592(31), 0
# CHECK-BE: xxmfacc 2                          # encoding: [0x7d,0x00,0x01,0x62]
# CHECK-LE: xxmfacc 2                          # encoding: [0x62,0x01,0x00,0x7d]
            xxmfacc 2
# CHECK-BE: xxmtacc 2                          # encoding: [0x7d,0x01,0x01,0x62]
# CHECK-LE: xxmtacc 2                          # encoding: [0x62,0x01,0x01,0x7d]
            xxmtacc 2
# CHECK-BE: xxsetaccz 1                        # encoding: [0x7c,0x83,0x01,0x62]
# CHECK-LE: xxsetaccz 1                        # encoding: [0x62,0x01,0x83,0x7c]
            xxsetaccz 1
# CHECK-BE: lxvp 2, 32(4)                      # encoding: [0x18,0x44,0x00,0x20]
# CHECK-LE: lxvp 2, 32(4)                      # encoding: [0x20,0x00,0x44,0x18]
            lxvp 2, 32(4)
# CHECK-BE: lxvp 34, 64(4)                     # encoding: [0x18,0x64,0x00,0x40]
# CHECK-LE: lxvp 34, 64(4)                     # encoding: [0x40,0x00,0x64,0x18]
            lxvp 34, 64(4)
# CHECK-BE: plxvp 2, -8589934592(0), 1         # encoding: [0x04,0x12,0x00,0x00,
# CHECK-BE-SAME:                                            0xe8,0x40,0x00,0x00]
# CHECK-LE: plxvp 2, -8589934592(0), 1         # encoding: [0x00,0x00,0x12,0x04,
# CHECK-LE-SAME:                                            0x00,0x00,0x40,0xe8]
            plxvp 2, -8589934592(0), 1
# CHECK-BE: plxvp 34, 8589934591(3), 0         # encoding: [0x04,0x01,0xff,0xff,
# CHECK-BE-SAME:                                            0xe8,0x63,0xff,0xff]
# CHECK-LE: plxvp 34, 8589934591(3), 0         # encoding: [0xff,0xff,0x01,0x04,
# CHECK-LE-SAME:                                            0xff,0xff,0x63,0xe8]
            plxvp 34, 8589934591(3), 0
# CHECK-BE: stxvp 2, 32(4)                     # encoding: [0x18,0x44,0x00,0x21]
# CHECK-LE: stxvp 2, 32(4)                     # encoding: [0x21,0x00,0x44,0x18]
            stxvp 2, 32(4)
# CHECK-BE: stxvp 34, 64(4)                    # encoding: [0x18,0x64,0x00,0x41]
# CHECK-LE: stxvp 34, 64(4)                    # encoding: [0x41,0x00,0x64,0x18]
            stxvp 34, 64(4)
# CHECK-BE: pstxvp 2, -8589934592(0), 1        # encoding: [0x04,0x12,0x00,0x00,
# CHECK-BE-SAME:                                            0xf8,0x40,0x00,0x00]
# CHECK-LE: pstxvp 2, -8589934592(0), 1        # encoding: [0x00,0x00,0x12,0x04
# CHECK-LE-SAME:                                            0x00,0x00,0x40,0xf8]
            pstxvp 2, -8589934592(0), 1
# CHECK-BE: pstxvp 34, 8589934591(3), 0        # encoding: [0x04,0x01,0xff,0xff
# CHECK-BE-SAME:                                            0xf8,0x63,0xff,0xff]
# CHECK-LE: pstxvp 34, 8589934591(3), 0        # encoding: [0xff,0xff,0x01,0x04
# CHECK-LE-SAME:                                            0xff,0xff,0x63,0xf8]
            pstxvp 34, 8589934591(3), 0
# CHECK-BE: lxvpx 2, 3, 4                      # encoding: [0x7c,0x43,0x22,0x9a]
# CHECK-LE: lxvpx 2, 3, 4                      # encoding: [0x9a,0x22,0x43,0x7c]
            lxvpx 2, 3, 4
# CHECK-BE: stxvpx 34, 6, 4                    # encoding: [0x7c,0x66,0x23,0x9a]
# CHECK-LE: stxvpx 34, 6, 4                    # encoding: [0x9a,0x23,0x66,0x7c]
            stxvpx 34, 6, 4
# CHECK-BE: paddi 1, 2, 8589934591, 0             # encoding: [0x06,0x01,0xff,0xff
# CHECK-BE-SAME:                                               0x38,0x22,0xff,0xff]
# CHECK-LE: paddi 1, 2, 8589934591, 0             # encoding: [0xff,0xff,0x01,0x06
# CHECK-LE-SAME:                                               0xff,0xff,0x22,0x38]
            paddi 1, 2, 8589934591, 0
# CHECK-BE: paddi 1, 0, -8589934592, 1            # encoding: [0x06,0x12,0x00,0x00
# CHECK-BE-SAME:                                               0x38,0x20,0x00,0x00]
# CHECK-LE: paddi 1, 0, -8589934592, 1            # encoding: [0x00,0x00,0x12,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x20,0x38]
            paddi 1, 0, -8589934592, 1
# CHECK-BE: pli 1, -8589934592                    # encoding: [0x06,0x02,0x00,0x00
# CHECK-BE-SAME:                                               0x38,0x20,0x00,0x00]
# CHECK-LE: pli 1, -8589934592                    # encoding: [0x00,0x00,0x02,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x20,0x38]
            pli 1, -8589934592
# CHECK-BE: pli 1, 8589934591                     # encoding: [0x06,0x01,0xff,0xff
# CHECK-BE-SAME:                                               0x38,0x20,0xff,0xff]
# CHECK-LE: pli 1, 8589934591                     # encoding: [0xff,0xff,0x01,0x06
# CHECK-LE-SAME:                                               0xff,0xff,0x20,0x38]
            pli 1, 8589934591
# CHECK-BE: pstfs 1, -134217728(3), 0             # encoding: [0x06,0x03,0xf8,0x00,
# CHECK-BE-SAME:                                               0xd0,0x23,0x00,0x00]
# CHECK-LE: pstfs 1, -134217728(3), 0             # encoding: [0x00,0xf8,0x03,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xd0]
            pstfs 1, -134217728(3), 0
# CHECK-BE: pstfs 1, 134217727(0), 1              # encoding: [0x06,0x10,0x07,0xff
# CHECK-BE-SAME:                                               0xd0,0x20,0xff,0xff]
# CHECK-LE: pstfs 1, 134217727(0), 1              # encoding: [0xff,0x07,0x10,0x06,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xd0]
            pstfs 1, 134217727(0), 1
# CHECK-BE: pstfd 1, -134217728(3), 0             # encoding: [0x06,0x03,0xf8,0x00,
# CHECK-BE-SAME:                                               0xd8,0x23,0x00,0x00]
# CHECK-LE: pstfd 1, -134217728(3), 0             # encoding: [0x00,0xf8,0x03,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xd8]
            pstfd 1, -134217728(3), 0
# CHECK-BE: pstfd 1, 134217727(0), 1              # encoding: [0x06,0x10,0x07,0xff
# CHECK-BE-SAME:                                               0xd8,0x20,0xff,0xff]
# CHECK-LE: pstfd 1, 134217727(0), 1              # encoding: [0xff,0x07,0x10,0x06,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xd8]
            pstfd 1, 134217727(0), 1
# CHECK-BE: pstxssp 1, -134217728(3), 0           # encoding: [0x04,0x03,0xf8,0x00,
# CHECK-BE-SAME:                                               0xbc,0x23,0x00,0x00]
# CHECK-LE: pstxssp 1, -134217728(3), 0           # encoding: [0x00,0xf8,0x03,0x04
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xbc]
            pstxssp 1, -134217728(3), 0
# CHECK-BE: pstxssp 1, 134217727(0), 1            # encoding: [0x04,0x10,0x07,0xff
# CHECK-BE-SAME:                                               0xbc,0x20,0xff,0xff]
# CHECK-LE: pstxssp 1, 134217727(0), 1            # encoding: [0xff,0x07,0x10,0x04,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xbc]
            pstxssp 1, 134217727(0), 1
# CHECK-BE: pstxsd 1, -134217728(3), 0            # encoding: [0x04,0x03,0xf8,0x00,
# CHECK-BE-SAME:                                               0xb8,0x23,0x00,0x00]
# CHECK-LE: pstxsd 1, -134217728(3), 0            # encoding: [0x00,0xf8,0x03,0x04
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xb8]
            pstxsd 1, -134217728(3), 0
# CHECK-BE: pstxsd 1, 134217727(0), 1             # encoding: [0x04,0x10,0x07,0xff
# CHECK-BE-SAME:                                               0xb8,0x20,0xff,0xff]
# CHECK-LE: pstxsd 1, 134217727(0), 1             # encoding: [0xff,0x07,0x10,0x04,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xb8]
            pstxsd 1, 134217727(0), 1
# CHECK-BE: plfs 1, -8589934592(3), 0             # encoding: [0x06,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0xc0,0x23,0x00,0x00]
# CHECK-LE: plfs 1, -8589934592(3), 0             # encoding: [0x00,0x00,0x02,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xc0]
            plfs 1, -8589934592(3), 0
# CHECK-BE: plfs 1, 8589934591(0), 1              # encoding: [0x06,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0xc0,0x20,0xff,0xff]
# CHECK-LE: plfs 1, 8589934591(0), 1              # encoding: [0xff,0xff,0x11,0x06,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xc0]
            plfs 1, 8589934591(0), 1
# CHECK-BE: plfd 1, -8589934592(3), 0             # encoding: [0x06,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0xc8,0x23,0x00,0x00]
# CHECK-LE: plfd 1, -8589934592(3), 0             # encoding: [0x00,0x00,0x02,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xc8]
            plfd 1, -8589934592(3), 0
# CHECK-BE: plfd 1, 8589934591(0), 1              # encoding: [0x06,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0xc8,0x20,0xff,0xff]
# CHECK-LE: plfd 1, 8589934591(0), 1              # encoding: [0xff,0xff,0x11,0x06,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xc8]
            plfd 1, 8589934591(0), 1
# CHECK-BE: plxssp 1, -8589934592(3), 0           # encoding: [0x04,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0xac,0x23,0x00,0x00]
# CHECK-LE: plxssp 1, -8589934592(3), 0           # encoding: [0x00,0x00,0x02,0x04
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xac]
            plxssp 1, -8589934592(3), 0
# CHECK-BE: plxssp 1, 8589934591(0), 1            # encoding: [0x04,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0xac,0x20,0xff,0xff]
# CHECK-LE: plxssp 1, 8589934591(0), 1            # encoding: [0xff,0xff,0x11,0x04,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xac]
            plxssp 1, 8589934591(0), 1
# CHECK-BE: plxsd 1, -8589934592(3), 0            # encoding: [0x04,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0xa8,0x23,0x00,0x00]
# CHECK-LE: plxsd 1, -8589934592(3), 0            # encoding: [0x00,0x00,0x02,0x04
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xa8]
            plxsd 1, -8589934592(3), 0
# CHECK-BE: plxsd 1, 8589934591(0), 1             # encoding: [0x04,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0xa8,0x20,0xff,0xff]
# CHECK-LE: plxsd 1, 8589934591(0), 1             # encoding: [0xff,0xff,0x11,0x04,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xa8]
            plxsd 1, 8589934591(0), 1
# CHECK-BE: pstb 1, -8589934592(3), 0             # encoding: [0x06,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0x98,0x23,0x00,0x00]
# CHECK-LE: pstb 1, -8589934592(3), 0             # encoding: [0x00,0x00,0x02,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0x98]
            pstb 1, -8589934592(3), 0
# CHECK-BE: pstb 1, 8589934591(0), 1              # encoding: [0x06,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0x98,0x20,0xff,0xff]
# CHECK-LE: pstb 1, 8589934591(0), 1              # encoding: [0xff,0xff,0x11,0x06,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0x98]
            pstb 1, 8589934591(0), 1
# CHECK-BE: psth 1, -8589934592(3), 0             # encoding: [0x06,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0xb0,0x23,0x00,0x00]
# CHECK-LE: psth 1, -8589934592(3), 0             # encoding: [0x00,0x00,0x02,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xb0]
            psth 1, -8589934592(3), 0
# CHECK-BE: psth 1, 8589934591(0), 1              # encoding: [0x06,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0xb0,0x20,0xff,0xff]
# CHECK-LE: psth 1, 8589934591(0), 1              # encoding: [0xff,0xff,0x11,0x06,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xb0]
            psth 1, 8589934591(0), 1
# CHECK-BE: pstw 1, -8589934592(3), 0             # encoding: [0x06,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0x90,0x23,0x00,0x00]
# CHECK-LE: pstw 1, -8589934592(3), 0             # encoding: [0x00,0x00,0x02,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0x90]
            pstw 1, -8589934592(3), 0
# CHECK-BE: pstw 1, 8589934591(0), 1              # encoding: [0x06,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0x90,0x20,0xff,0xff]
# CHECK-LE: pstw 1, 8589934591(0), 1              # encoding: [0xff,0xff,0x11,0x06,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0x90]
            pstw 1, 8589934591(0), 1
# CHECK-BE: pstd 1, -8589934592(3), 0             # encoding: [0x04,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0xf4,0x23,0x00,0x00]
# CHECK-LE: pstd 1, -8589934592(3), 0             # encoding: [0x00,0x00,0x02,0x04
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xf4]
            pstd 1, -8589934592(3), 0
# CHECK-BE: pstd 1, 8589934591(0), 1              # encoding: [0x04,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0xf4,0x20,0xff,0xff]
# CHECK-LE: pstd 1, 8589934591(0), 1              # encoding: [0xff,0xff,0x11,0x04,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xf4]
            pstd 1, 8589934591(0), 1
# CHECK-BE: plbz 1, 8589934591(3), 0              # encoding: [0x06,0x01,0xff,0xff
# CHECK-BE-SAME:                                               0x88,0x23,0xff,0xff]
# CHECK-LE: plbz 1, 8589934591(3), 0              # encoding: [0xff,0xff,0x01,0x06
# CHECK-LE-SAME:                                               0xff,0xff,0x23,0x88]
            plbz 1, 8589934591(3), 0
# CHECK-BE: plbz 1, -8589934592(0), 1             # encoding: [0x06,0x12,0x00,0x00
# CHECK-BE-SAME:                                               0x88,0x20,0x00,0x00]
# CHECK-LE: plbz 1, -8589934592(0), 1             # encoding: [0x00,0x00,0x12,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x20,0x88]
            plbz 1, -8589934592(0), 1
# CHECK-BE: plhz 1, 8589934591(3), 0              # encoding: [0x06,0x01,0xff,0xff
# CHECK-BE-SAME:                                               0xa0,0x23,0xff,0xff]
# CHECK-LE: plhz 1, 8589934591(3), 0              # encoding: [0xff,0xff,0x01,0x06
# CHECK-LE-SAME:                                               0xff,0xff,0x23,0xa0]
            plhz 1, 8589934591(3), 0
# CHECK-BE: plhz 1, -8589934592(0), 1             # encoding: [0x06,0x12,0x00,0x00
# CHECK-BE-SAME:                                               0xa0,0x20,0x00,0x00]
# CHECK-LE: plhz 1, -8589934592(0), 1             # encoding: [0x00,0x00,0x12,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x20,0xa0]
            plhz 1, -8589934592(0), 1
# CHECK-BE: plha 1, 8589934591(3), 0              # encoding: [0x06,0x01,0xff,0xff
# CHECK-BE-SAME:                                               0xa8,0x23,0xff,0xff]
# CHECK-LE: plha 1, 8589934591(3), 0              # encoding: [0xff,0xff,0x01,0x06
# CHECK-LE-SAME:                                               0xff,0xff,0x23,0xa8]
            plha 1, 8589934591(3), 0
# CHECK-BE: plha 1, -8589934592(0), 1             # encoding: [0x06,0x12,0x00,0x00
# CHECK-BE-SAME:                                               0xa8,0x20,0x00,0x00]
# CHECK-LE: plha 1, -8589934592(0), 1             # encoding: [0x00,0x00,0x12,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x20,0xa8]
            plha 1, -8589934592(0), 1
# CHECK-BE: plwz 1, 8589934591(3), 0              # encoding: [0x06,0x01,0xff,0xff
# CHECK-BE-SAME:                                               0x80,0x23,0xff,0xff]
# CHECK-LE: plwz 1, 8589934591(3), 0              # encoding: [0xff,0xff,0x01,0x06
# CHECK-LE-SAME:                                               0xff,0xff,0x23,0x80]
            plwz 1, 8589934591(3), 0
# CHECK-BE: plwz 1, -8589934592(0), 1             # encoding: [0x06,0x12,0x00,0x00
# CHECK-BE-SAME:                                               0x80,0x20,0x00,0x00]
# CHECK-LE: plwz 1, -8589934592(0), 1             # encoding: [0x00,0x00,0x12,0x06
# CHECK-LE-SAME:                                               0x00,0x00,0x20,0x80]
            plwz 1, -8589934592(0), 1
# CHECK-BE: plwa 1, -8589934592(3), 0             # encoding: [0x04,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0xa4,0x23,0x00,0x00]
# CHECK-LE: plwa 1, -8589934592(3), 0             # encoding: [0x00,0x00,0x02,0x04
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xa4]
            plwa 1, -8589934592(3), 0
# CHECK-BE: plwa 1, 8589934591(0), 1              # encoding: [0x04,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0xa4,0x20,0xff,0xff]
# CHECK-LE: plwa 1, 8589934591(0), 1              # encoding: [0xff,0xff,0x11,0x04,
# CECHK-LE-SAME:                                               0xff,0xff,0x20,0xa4]
            plwa 1, 8589934591(0), 1
# CHECK-BE: pld 1, -8589934592(3), 0              # encoding: [0x04,0x02,0x00,0x00,
# CHECK-BE-SAME:                                               0xe4,0x23,0x00,0x00]
# CHECK-LE: pld 1, -8589934592(3), 0              # encoding: [0x00,0x00,0x02,0x04
# CHECK-LE-SAME:                                               0x00,0x00,0x23,0xe4]
            pld 1, -8589934592(3), 0
# CHECK-BE: pld 1, 8589934591(0), 1               # encoding: [0x04,0x11,0xff,0xff
# CHECK-BE-SAME:                                               0xe4,0x20,0xff,0xff]
# CHECK-LE: pld 1, 8589934591(0), 1               # encoding: [0xff,0xff,0x11,0x04,
# CHECK-LE-SAME:                                               0xff,0xff,0x20,0xe4]
            pld 1, 8589934591(0), 1
# CHECK-BE: vpdepd 1, 2, 0                        # encoding: [0x10,0x22,0x05,0xcd]
# CHECK-LE: vpdepd 1, 2, 0                        # encoding: [0xcd,0x05,0x22,0x10]
            vpdepd 1, 2, 0
# CHECK-BE: vpextd 1, 2, 0                        # encoding: [0x10,0x22,0x05,0x8d]
# CHECK-LE: vpextd 1, 2, 0                        # encoding: [0x8d,0x05,0x22,0x10]
            vpextd 1, 2, 0
# CHECK-BE: pdepd 1, 2, 4                         # encoding: [0x7c,0x41,0x21,0x38]
# CHECK-LE: pdepd 1, 2, 4                         # encoding: [0x38,0x21,0x41,0x7c]
            pdepd 1, 2, 4
# CHECK-BE: pextd 1, 2, 4                         # encoding: [0x7c,0x41,0x21,0x78]
# CHECK-LE: pextd 1, 2, 4                         # encoding: [0x78,0x21,0x41,0x7c]
            pextd 1, 2, 4
# CHECK-BE: vcfuged 1, 2, 4                       # encoding: [0x10,0x22,0x25,0x4d]
# CHECK-LE: vcfuged 1, 2, 4                       # encoding: [0x4d,0x25,0x22,0x10]
            vcfuged 1, 2, 4
# CHECK-BE: cfuged 1, 2, 4                        # encoding: [0x7c,0x41,0x21,0xb8]
# CHECK-LE: cfuged 1, 2, 4                        # encoding: [0xb8,0x21,0x41,0x7c]
            cfuged 1, 2, 4
# CHECK-BE: vgnb 1, 2, 2                          # encoding: [0x10,0x22,0x14,0xcc]
# CHECK-LE: vgnb 1, 2, 2                          # encoding: [0xcc,0x14,0x22,0x10]
            vgnb 1, 2, 2
# CHECK-BE: xxeval 32, 1, 2, 3, 2                 # encoding: [0x05,0x00,0x00,0x02,
# CHECK-BE-SAME:                                               0x88,0x01,0x10,0xd1]
# CHECK-LE: xxeval 32, 1, 2, 3, 2                 # encoding: [0x02,0x00,0x00,0x05,
# CHECK-LE-SAME:                                               0xd1,0x10,0x01,0x88]
            xxeval 32, 1, 2, 3, 2
# CHECK-BE: vclzdm 1, 2, 3                        # encoding: [0x10,0x22,0x1f,0x84]
# CHECK-LE: vclzdm 1, 2, 3                        # encoding: [0x84,0x1f,0x22,0x10]
            vclzdm 1, 2, 3
# CHECK-BE: vctzdm 1, 2, 3                        # encoding: [0x10,0x22,0x1f,0xc4]
# CHECK-LE: vctzdm 1, 2, 3                        # encoding: [0xc4,0x1f,0x22,0x10]
            vctzdm 1, 2, 3
# CHECK-BE: cntlzdm 1, 3, 2                       # encoding: [0x7c,0x61,0x10,0x76]
# CHECK-LE: cntlzdm 1, 3, 2                       # encoding: [0x76,0x10,0x61,0x7c]
            cntlzdm 1, 3, 2
# CHECK-BE: cnttzdm 1, 3, 2                       # encoding: [0x7c,0x61,0x14,0x76]
# CHECK-LE: cnttzdm 1, 3, 2                       # encoding: [0x76,0x14,0x61,0x7c]
            cnttzdm 1, 3, 2
# CHECK-BE: xxgenpcvbm 0, 1, 2                    # encoding: [0xf0,0x02,0x0f,0x28]
# CHECK-LE: xxgenpcvbm 0, 1, 2                    # encoding: [0x28,0x0f,0x02,0xf0]
            xxgenpcvbm 0, 1, 2
# CHECK-BE: xxgenpcvhm 0, 1, 2                    # encoding: [0xf0,0x02,0x0f,0x2a]
# CHECK-LE: xxgenpcvhm 0, 1, 2                    # encoding: [0x2a,0x0f,0x02,0xf0]
            xxgenpcvhm 0, 1, 2
# CHECK-BE: xxgenpcvwm 0, 1, 2                    # encoding: [0xf0,0x02,0x0f,0x68]
# CHECK-LE: xxgenpcvwm 0, 1, 2                    # encoding: [0x68,0x0f,0x02,0xf0]
            xxgenpcvwm 0, 1, 2
# CHECK-BE: xxgenpcvdm 0, 1, 2                    # encoding: [0xf0,0x02,0x0f,0x6a]
# CHECK-LE: xxgenpcvdm 0, 1, 2                    # encoding: [0x6a,0x0f,0x02,0xf0]
            xxgenpcvdm 0, 1, 2
# CHECK-BE: vclrlb 1, 4, 3                        # encoding: [0x10,0x24,0x19,0x8d]
# CHECK-LE: vclrlb 1, 4, 3                        # encoding: [0x8d,0x19,0x24,0x10]
            vclrlb 1, 4, 3
# CHECK-BE: vclrrb 1, 4, 3                        # encoding: [0x10,0x24,0x19,0xcd]
# CHECK-LE: vclrrb 1, 4, 3                        # encoding: [0xcd,0x19,0x24,0x10]
            vclrrb 1, 4, 3
# Boundary conditions of 8RR_DForm_IMM32_XT6's immediates
# CHECK-BE: xxspltiw 63, 4294901760               # encoding: [0x05,0x00,0xff,0xff,
# CHECK-BE-SAME:                                               0x83,0xe7,0x00,0x00]
# CHECK-LE: xxspltiw 63, 4294901760               # encoding: [0xff,0xff,0x00,0x05,
# CHECK-LE-SAME:                                               0x00,0x00,0xe7,0x83]
            xxspltiw 63, 4294901760
# CHECK-BE: xxspltiw 63, 65535                    # encoding: [0x05,0x00,0x00,0x00,
# CHECK-BE-SAME:                                               0x83,0xe7,0xff,0xff]
# CHECK-LE: xxspltiw 63, 65535                    # encoding: [0x00,0x00,0x00,0x05,
# CHECK-LE-SAME:                                               0xff,0xff,0xe7,0x83]
            xxspltiw 63, 65535
# CHECK-BE: xxspltiw 63, 4294967295               # encoding: [0x05,0x00,0xff,0xff,
# CHECK-BE-SAME:                                               0x83,0xe7,0xff,0xff]
# CHECK-LE: xxspltiw 63, 4294967295               # encoding: [0xff,0xff,0x00,0x05,
# CHECK-LE-SAME:                                               0xff,0xff,0xe7,0x83]
            xxspltiw 63, 4294967295
# CHECK-BE: xxspltiw 63, -1                       # encoding: [0x05,0x00,0xff,0xff,
# CHECK-BE-SAME:                                               0x83,0xe7,0xff,0xff]
# CHECK-LE: xxspltiw 63, -1                       # encoding: [0xff,0xff,0x00,0x05,
# CHECK-LE-SAME:                                               0xff,0xff,0xe7,0x83]
            xxspltiw 63, -1
# CHECK-BE: xxspltidp 63, 4294967295              # encoding: [0x05,0x00,0xff,0xff,
# CHECK-BE-SAME:                                               0x83,0xe5,0xff,0xff]
# CHECK-LE: xxspltidp 63, 4294967295              # encoding: [0xff,0xff,0x00,0x05,
# CHECK-LE-SAME:                                               0xff,0xff,0xe5,0x83]
            xxspltidp 63, 4294967295
# Boundary conditions of 8RR_DForm_IMM32_XT6_IX's immediates
# CHECK-BE: xxsplti32dx 63, 1, 4294901760         # encoding: [0x05,0x00,0xff,0xff,
# CHECK-BE-SAME:                                               0x83,0xe3,0x00,0x00]
# CHECK-LE: xxsplti32dx 63, 1, 4294901760         # encoding: [0xff,0xff,0x00,0x05,
# CHECK-LE-SAME:                                               0x00,0x00,0xe3,0x83]
            xxsplti32dx 63, 1, 4294901760
# CHECK-BE: xxsplti32dx 63, 1, 65535              # encoding: [0x05,0x00,0x00,0x00,
# CHECK-BE-SAME:                                               0x83,0xe3,0xff,0xff]
# CHECK-LE: xxsplti32dx 63, 1, 65535              # encoding: [0x00,0x00,0x00,0x05,
# CHECK-LE-SAME:                                               0xff,0xff,0xe3,0x83]
            xxsplti32dx 63, 1, 65535
# CHECK-BE: xxsplti32dx 63, 1, 4294967295         # encoding: [0x05,0x00,0xff,0xff,
# CHECK-BE-SAME:                                               0x83,0xe3,0xff,0xff]
# CHECK-LE: xxsplti32dx 63, 1, 4294967295         # encoding: [0xff,0xff,0x00,0x05,
# CHECK-LE-SAME:                                               0xff,0xff,0xe3,0x83]
            xxsplti32dx 63, 1, 4294967295
# CHECK-BE: xxsplti32dx 63, 1, -1                 # encoding: [0x05,0x00,0xff,0xff,
# CHECK-BE-SAME:                                               0x83,0xe3,0xff,0xff]
# CHECK-LE: xxsplti32dx 63, 1, -1                 # encoding: [0xff,0xff,0x00,0x05,
# CHECK-LE-SAME:                                               0xff,0xff,0xe3,0x83]
            xxsplti32dx 63, 1, -1
# CHECK-BE: xxpermx 6, 63, 21, 34, 2              # encoding: [0x05,0x00,0x00,0x02,
# CHECK-BE-SAME:                                               0x88,0xdf,0xa8,0x8c]
# CHECK-LE: xxpermx 6, 63, 21, 34, 2              # encoding: [0x02,0x00,0x00,0x05,
# CHECK-LE-SAME:                                               0x8c,0xa8,0xdf,0x88]
            xxpermx 6, 63, 21, 34, 2
# CHECK-BE: xxblendvb 6, 63, 21, 34               # encoding: [0x05,0x00,0x00,0x00,
# CHECK-BE-SAME:                                               0x84,0xdf,0xa8,0x8c]
# CHECK-LE: xxblendvb 6, 63, 21, 34               # encoding: [0x00,0x00,0x00,0x05,
# CHECK-LE-SAME:                                               0x8c,0xa8,0xdf,0x84]
            xxblendvb 6, 63, 21, 34
# CHECK-BE: xxblendvh 6, 63, 21, 34               # encoding: [0x05,0x00,0x00,0x00,
# CHECK-BE-SAME:                                               0x84,0xdf,0xa8,0x9c]
# CHECK-LE: xxblendvh 6, 63, 21, 34               # encoding: [0x00,0x00,0x00,0x05,
# CHECK-LE-SAME:                                               0x9c,0xa8,0xdf,0x84]
            xxblendvh 6, 63, 21, 34
# CHECK-BE: xxblendvw 6, 63, 21, 34               # encoding: [0x05,0x00,0x00,0x00,
# CHECK-BE-SAME:                                               0x84,0xdf,0xa8,0xac]
# CHECK-LE: xxblendvw 6, 63, 21, 34               # encoding: [0x00,0x00,0x00,0x05,
# CHECK-LE-SAME:                                               0xac,0xa8,0xdf,0x84]
            xxblendvw 6, 63, 21, 34
# CHECK-BE: xxblendvd 6, 63, 21, 34               # encoding: [0x05,0x00,0x00,0x00,
# CHECK-BE-SAME:                                               0x84,0xdf,0xa8,0xbc]
# CHECK-LE: xxblendvd 6, 63, 21, 34               # encoding: [0x00,0x00,0x00,0x05,
# CHECK-LE-SAME:                                               0xbc,0xa8,0xdf,0x84]
            xxblendvd 6, 63, 21, 34
# CHECK-BE: setbc 21, 11                          # encoding: [0x7e,0xab,0x03,0x00]
# CHECK-LE: setbc 21, 11                          # encoding: [0x00,0x03,0xab,0x7e]
            setbc 21, 11
# CHECK-BE: setbcr 21, 11                         # encoding: [0x7e,0xab,0x03,0x40]
# CHECK-LE: setbcr 21, 11                         # encoding: [0x40,0x03,0xab,0x7e]
            setbcr 21, 11
# CHECK-BE: setnbc 21, 11                         # encoding: [0x7e,0xab,0x03,0x80]
# CHECK-LE: setnbc 21, 11                         # encoding: [0x80,0x03,0xab,0x7e]
            setnbc 21, 11
# CHECK-BE: setnbcr 21, 11                        # encoding: [0x7e,0xab,0x03,0xc0]
# CHECK-LE: setnbcr 21, 11                        # encoding: [0xc0,0x03,0xab,0x7e]
            setnbcr 21, 11
# CHECK-BE: vsldbi 2, 3, 4, 5                     # encoding: [0x10,0x43,0x21,0x56]
# CHECK-LE: vsldbi 2, 3, 4, 5                     # encoding: [0x56,0x21,0x43,0x10]
            vsldbi 2, 3, 4, 5
# CHECK-BE: vsrdbi 2, 3, 4, 5                     # encoding: [0x10,0x43,0x23,0x56]
# CHECK-LE: vsrdbi 2, 3, 4, 5                     # encoding: [0x56,0x23,0x43,0x10]
            vsrdbi 2, 3, 4, 5
# CHECK-BE: vmulld 1, 2, 3                        # encoding: [0x10,0x22,0x19,0xc9]
# CHECK-LE: vmulld 1, 2, 3                        # encoding: [0xc9,0x19,0x22,0x10]
            vmulld 1, 2, 3
# CHECK-BE: vmulhsw 1, 2, 3                       # encoding: [0x10,0x22,0x1b,0x89]
# CHECK-LE: vmulhsw 1, 2, 3                       # encoding: [0x89,0x1b,0x22,0x10]
            vmulhsw 1, 2, 3
# CHECK-BE: vmulhuw 1, 2, 3                       # encoding: [0x10,0x22,0x1a,0x89]
# CHECK-LE: vmulhuw 1, 2, 3                       # encoding: [0x89,0x1a,0x22,0x10]
            vmulhuw 1, 2, 3
# CHECK-BE: vmulhsd 1, 2, 3                       # encoding: [0x10,0x22,0x1b,0xc9]
# CHECK-LE: vmulhsd 1, 2, 3                       # encoding: [0xc9,0x1b,0x22,0x10]
            vmulhsd 1, 2, 3
# CHECK-BE: vmulhud 1, 2, 3                       # encoding: [0x10,0x22,0x1a,0xc9]
# CHECK-LE: vmulhud 1, 2, 3                       # encoding: [0xc9,0x1a,0x22,0x10]
            vmulhud 1, 2, 3
# CHECK-BE: vmodsw 21, 11, 10                     # encoding: [0x12,0xab,0x57,0x8b]
# CHECK-LE: vmodsw 21, 11, 10                     # encoding: [0x8b,0x57,0xab,0x12]
            vmodsw 21, 11, 10
# CHECK-BE: vmoduw 21, 11, 10                     # encoding: [0x12,0xab,0x56,0x8b]
# CHECK-LE: vmoduw 21, 11, 10                     # encoding: [0x8b,0x56,0xab,0x12]
            vmoduw 21, 11, 10
# CHECK-BE: vmodsd 21, 11, 10                     # encoding: [0x12,0xab,0x57,0xcb]
# CHECK-LE: vmodsd 21, 11, 10                     # encoding: [0xcb,0x57,0xab,0x12]
            vmodsd 21, 11, 10
# CHECK-BE: vmodud 21, 11, 10                     # encoding: [0x12,0xab,0x56,0xcb]
# CHECK-LE: vmodud 21, 11, 10                     # encoding: [0xcb,0x56,0xab,0x12]
            vmodud 21, 11, 10
# CHECK-BE: vdivsw 21, 11, 10                     # encoding: [0x12,0xab,0x51,0x8b]
# CHECK-LE: vdivsw 21, 11, 10                     # encoding: [0x8b,0x51,0xab,0x12]
            vdivsw 21, 11, 10
# CHECK-BE: vdivuw 21, 11, 10                     # encoding: [0x12,0xab,0x50,0x8b]
# CHECK-LE: vdivuw 21, 11, 10                     # encoding: [0x8b,0x50,0xab,0x12]
            vdivuw 21, 11, 10
# CHECK-BE: vdivsd 21, 11, 10                     # encoding: [0x12,0xab,0x51,0xcb]
# CHECK-LE: vdivsd 21, 11, 10                     # encoding: [0xcb,0x51,0xab,0x12]
            vdivsd 21, 11, 10
# CHECK-BE: vdivud 21, 11, 10                     # encoding: [0x12,0xab,0x50,0xcb]
# CHECK-LE: vdivud 21, 11, 10                     # encoding: [0xcb,0x50,0xab,0x12]
            vdivud 21, 11, 10
# CHECK-BE: vdivesw 21, 11, 10                    # encoding: [0x12,0xab,0x53,0x8b]
# CHECK-LE: vdivesw 21, 11, 10                    # encoding: [0x8b,0x53,0xab,0x12]
            vdivesw 21, 11, 10
# CHECK-BE: vdiveuw 21, 11, 10                    # encoding: [0x12,0xab,0x52,0x8b]
# CHECK-LE: vdiveuw 21, 11, 10                    # encoding: [0x8b,0x52,0xab,0x12]
            vdiveuw 21, 11, 10
# CHECK-BE: vdivesd 21, 11, 10                    # encoding: [0x12,0xab,0x53,0xcb]
# CHECK-LE: vdivesd 21, 11, 10                    # encoding: [0xcb,0x53,0xab,0x12]
            vdivesd 21, 11, 10
# CHECK-BE: vdiveud 21, 11, 10                    # encoding: [0x12,0xab,0x52,0xcb]
# CHECK-LE: vdiveud 21, 11, 10                    # encoding: [0xcb,0x52,0xab,0x12]
            vdiveud 21, 11, 10
# CHECK-BE: vinsw 2, 3, 12                        # encoding: [0x10,0x4c,0x18,0xcf]
# CHECK-LE: vinsw 2, 3, 12                        # encoding: [0xcf,0x18,0x4c,0x10]
            vinsw 2, 3, 12
# CHECK-BE: vinsd 2, 3, 12                        # encoding: [0x10,0x4c,0x19,0xcf]
# CHECK-LE: vinsd 2, 3, 12                        # encoding: [0xcf,0x19,0x4c,0x10]
            vinsd 2, 3, 12
# CHECK-BE: vinsbvlx 1, 3, 5                      # encoding: [0x10,0x23,0x28,0x0f]
# CHECK-LE: vinsbvlx 1, 3, 5                      # encoding: [0x0f,0x28,0x23,0x10]
            vinsbvlx 1, 3, 5
# CHECK-BE: vinsbvrx 1, 3, 5                      # encoding: [0x10,0x23,0x29,0x0f]
# CHECK-LE: vinsbvrx 1, 3, 5                      # encoding: [0x0f,0x29,0x23,0x10]
            vinsbvrx 1, 3, 5
# CHECK-BE: vinshvlx 1, 3, 5                      # encoding: [0x10,0x23,0x28,0x4f]
# CHECK-LE: vinshvlx 1, 3, 5                      # encoding: [0x4f,0x28,0x23,0x10]
            vinshvlx 1, 3, 5
# CHECK-BE: vinshvrx 1, 3, 5                      # encoding: [0x10,0x23,0x29,0x4f]
# CHECK-LE: vinshvrx 1, 3, 5                      # encoding: [0x4f,0x29,0x23,0x10]
            vinshvrx 1, 3, 5
# CHECK-BE: vinswvlx 1, 3, 5                      # encoding: [0x10,0x23,0x28,0x8f]
# CHECK-LE: vinswvlx 1, 3, 5                      # encoding: [0x8f,0x28,0x23,0x10]
            vinswvlx 1, 3, 5
# CHECK-BE: vinswvrx 1, 3, 5                      # encoding: [0x10,0x23,0x29,0x8f]
# CHECK-LE: vinswvrx 1, 3, 5                      # encoding: [0x8f,0x29,0x23,0x10]
            vinswvrx 1, 3, 5
# CHECK-BE: vinsblx 1, 2, 3                       # encoding: [0x10,0x22,0x1a,0x0f]
# CHECK-LE: vinsblx 1, 2, 3                       # encoding: [0x0f,0x1a,0x22,0x10]
            vinsblx 1, 2, 3
# CHECK-BE: vinsbrx 1, 2, 3                       # encoding: [0x10,0x22,0x1b,0x0f]
# CHECK-LE: vinsbrx 1, 2, 3                       # encoding: [0x0f,0x1b,0x22,0x10]
            vinsbrx 1, 2, 3
# CHECK-BE: vinshlx 1, 2, 3                       # encoding: [0x10,0x22,0x1a,0x4f]
# CHECK-LE: vinshlx 1, 2, 3                       # encoding: [0x4f,0x1a,0x22,0x10]
            vinshlx 1, 2, 3
# CHECK-BE: vinshrx 1, 2, 3                       # encoding: [0x10,0x22,0x1b,0x4f]
# CHECK-LE: vinshrx 1, 2, 3                       # encoding: [0x4f,0x1b,0x22,0x10]
            vinshrx 1, 2, 3
# CHECK-BE: vinswlx 1, 2, 3                       # encoding: [0x10,0x22,0x1a,0x8f]
# CHECK-LE: vinswlx 1, 2, 3                       # encoding: [0x8f,0x1a,0x22,0x10]
            vinswlx 1, 2, 3
# CHECK-BE: vinswrx 1, 2, 3                       # encoding: [0x10,0x22,0x1b,0x8f]
# CHECK-LE: vinswrx 1, 2, 3                       # encoding: [0x8f,0x1b,0x22,0x10]
            vinswrx 1, 2, 3
# CHECK-BE: vinsdlx 1, 2, 3                       # encoding: [0x10,0x22,0x1a,0xcf]
# CHECK-LE: vinsdlx 1, 2, 3                       # encoding: [0xcf,0x1a,0x22,0x10]
            vinsdlx 1, 2, 3
# CHECK-BE: vinsdrx 1, 2, 3                       # encoding: [0x10,0x22,0x1b,0xcf]
# CHECK-LE: vinsdrx 1, 2, 3                       # encoding: [0xcf,0x1b,0x22,0x10]
            vinsdrx 1, 2, 3
# CHECK-BE: vextdubvlx 1, 2, 3, 3                 # encoding: [0x10,0x22,0x18,0xd8]
# CHECK-LE: vextdubvlx 1, 2, 3, 3                 # encoding: [0xd8,0x18,0x22,0x10]
            vextdubvlx 1, 2, 3, 3
# CHECK-BE: vextdubvrx 1, 2, 3, 3                 # encoding: [0x10,0x22,0x18,0xd9]
# CHECK-LE: vextdubvrx 1, 2, 3, 3                 # encoding: [0xd9,0x18,0x22,0x10]
            vextdubvrx 1, 2, 3, 3
# CHECK-BE: vextduhvlx 1, 2, 3, 3                 # encoding: [0x10,0x22,0x18,0xda]
# CHECK-LE: vextduhvlx 1, 2, 3, 3                 # encoding: [0xda,0x18,0x22,0x10]
            vextduhvlx 1, 2, 3, 3
# CHECK-BE: vextduhvrx 1, 2, 3, 3                 # encoding: [0x10,0x22,0x18,0xdb]
# CHECK-LE: vextduhvrx 1, 2, 3, 3                 # encoding: [0xdb,0x18,0x22,0x10]
            vextduhvrx 1, 2, 3, 3
# CHECK-BE: vextduwvlx 1, 2, 3, 3                 # encoding: [0x10,0x22,0x18,0xdc]
# CHECK-LE: vextduwvlx 1, 2, 3, 3                 # encoding: [0xdc,0x18,0x22,0x10]
            vextduwvlx 1, 2, 3, 3
# CHECK-BE: vextduwvrx 1, 2, 3, 3                 # encoding: [0x10,0x22,0x18,0xdd]
# CHECK-LE: vextduwvrx 1, 2, 3, 3                 # encoding: [0xdd,0x18,0x22,0x10]
            vextduwvrx 1, 2, 3, 3
# CHECK-BE: vextddvlx 1, 2, 3, 3                  # encoding: [0x10,0x22,0x18,0xde]
# CHECK-LE: vextddvlx 1, 2, 3, 3                  # encoding: [0xde,0x18,0x22,0x10]
            vextddvlx 1, 2, 3, 3
# CHECK-BE: vextddvrx 1, 2, 3, 3                  # encoding: [0x10,0x22,0x18,0xdf]
# CHECK-LE: vextddvrx 1, 2, 3, 3                  # encoding: [0xdf,0x18,0x22,0x10]
            vextddvrx 1, 2, 3, 3
# CHECK-BE: lxvrbx 32, 1, 2                       # encoding: [0x7c,0x01,0x10,0x1b]
# CHECK-LE: lxvrbx 32, 1, 2                       # encoding: [0x1b,0x10,0x01,0x7c]
            lxvrbx 32, 1, 2
# CHECK-BE: lxvrhx 33, 1, 2                       # encoding: [0x7c,0x21,0x10,0x5b]
# CHECK-LE: lxvrhx 33, 1, 2                       # encoding: [0x5b,0x10,0x21,0x7c]
            lxvrhx 33, 1, 2
# CHECK-BE: lxvrdx 34, 1, 2                       # encoding: [0x7c,0x41,0x10,0xdb]
# CHECK-LE: lxvrdx 34, 1, 2                       # encoding: [0xdb,0x10,0x41,0x7c]
            lxvrdx 34, 1, 2
# CHECK-BE: lxvrwx 35, 1, 2                       # encoding: [0x7c,0x61,0x10,0x9b]
# CHECK-LE: lxvrwx 35, 1, 2                       # encoding: [0x9b,0x10,0x61,0x7c]
            lxvrwx 35, 1, 2
# CHECK-BE: stxvrbx 32, 3, 1                      # encoding: [0x7c,0x03,0x09,0x1b]
# CHECK-LE: stxvrbx 32, 3, 1                      # encoding: [0x1b,0x09,0x03,0x7c]
            stxvrbx 32, 3, 1
# CHECK-BE: stxvrhx 33, 3, 1                      # encoding: [0x7c,0x23,0x09,0x5b]
# CHECK-LE: stxvrhx 33, 3, 1                      # encoding: [0x5b,0x09,0x23,0x7c]
            stxvrhx 33, 3, 1
# CHECK-BE: stxvrwx 34, 3, 1                      # encoding: [0x7c,0x43,0x09,0x9b]
# CHECK-LE: stxvrwx 34, 3, 1                      # encoding: [0x9b,0x09,0x43,0x7c]
            stxvrwx 34, 3, 1
# CHECK-BE: stxvrdx 35, 3, 1                      # encoding: [0x7c,0x63,0x09,0xdb]
# CHECK-LE: stxvrdx 35, 3, 1                      # encoding: [0xdb,0x09,0x63,0x7c]
            stxvrdx 35, 3, 1
# CHECK-BE: vextractbm 1, 2                       # encoding: [0x10,0x28,0x16,0x42]
# CHECK-LE: vextractbm 1, 2                       # encoding: [0x42,0x16,0x28,0x10]
            vextractbm 1, 2
# CHECK-BE: vextracthm 1, 2                       # encoding: [0x10,0x29,0x16,0x42]
# CHECK-LE: vextracthm 1, 2                       # encoding: [0x42,0x16,0x29,0x10]
            vextracthm 1, 2
# CHECK-BE: vextractwm 1, 2                       # encoding: [0x10,0x2a,0x16,0x42]
# CHECK-LE: vextractwm 1, 2                       # encoding: [0x42,0x16,0x2a,0x10]
            vextractwm 1, 2
# CHECK-BE: vextractdm 1, 2                       # encoding: [0x10,0x2b,0x16,0x42]
# CHECK-LE: vextractdm 1, 2                       # encoding: [0x42,0x16,0x2b,0x10]
            vextractdm 1, 2
# CHECK-BE: vextractqm 1, 2                       # encoding: [0x10,0x2c,0x16,0x42]
# CHECK-LE: vextractqm 1, 2                       # encoding: [0x42,0x16,0x2c,0x10]
            vextractqm 1, 2
# CHECK-BE: vexpandbm 1, 2                        # encoding: [0x10,0x20,0x16,0x42]
# CHECK-LE: vexpandbm 1, 2                        # encoding: [0x42,0x16,0x20,0x10]
            vexpandbm 1, 2
# CHECK-BE: vexpandhm 1, 2                        # encoding: [0x10,0x21,0x16,0x42]
# CHECK-LE: vexpandhm 1, 2                        # encoding: [0x42,0x16,0x21,0x10]
            vexpandhm 1, 2
# CHECK-BE: vexpandwm 1, 2                        # encoding: [0x10,0x22,0x16,0x42]
# CHECK-LE: vexpandwm 1, 2                        # encoding: [0x42,0x16,0x22,0x10]
            vexpandwm 1, 2
# CHECK-BE: vexpanddm 1, 2                        # encoding: [0x10,0x23,0x16,0x42]
# CHECK-LE: vexpanddm 1, 2                        # encoding: [0x42,0x16,0x23,0x10]
            vexpanddm 1, 2
# CHECK-BE: vexpandqm 1, 2                        # encoding: [0x10,0x24,0x16,0x42]
# CHECK-LE: vexpandqm 1, 2                        # encoding: [0x42,0x16,0x24,0x10]
            vexpandqm 1, 2
# CHECK-BE: mtvsrbm 1, 2                          # encoding: [0x10,0x30,0x16,0x42]
# CHECK-LE: mtvsrbm 1, 2                          # encoding: [0x42,0x16,0x30,0x10]
            mtvsrbm 1, 2
# CHECK-BE: mtvsrhm 1, 2                          # encoding: [0x10,0x31,0x16,0x42]
# CHECK-LE: mtvsrhm 1, 2                          # encoding: [0x42,0x16,0x31,0x10]
            mtvsrhm 1, 2
# CHECK-BE: mtvsrwm 1, 2                          # encoding: [0x10,0x32,0x16,0x42]
# CHECK-LE: mtvsrwm 1, 2                          # encoding: [0x42,0x16,0x32,0x10]
            mtvsrwm 1, 2
# CHECK-BE: mtvsrdm 1, 2                          # encoding: [0x10,0x33,0x16,0x42]
# CHECK-LE: mtvsrdm 1, 2                          # encoding: [0x42,0x16,0x33,0x10]
            mtvsrdm 1, 2
# CHECK-BE: mtvsrqm 1, 2                          # encoding: [0x10,0x34,0x16,0x42]
# CHECK-LE: mtvsrqm 1, 2                          # encoding: [0x42,0x16,0x34,0x10]
            mtvsrqm 1, 2
# CHECK-BE: mtvsrbmi 1, 31                        # encoding: [0x10,0x2f,0x00,0x15]
# CHECK-LE: mtvsrbmi 1, 31                        # encoding: [0x15,0x00,0x2f,0x10]
            mtvsrbmi 1, 31
# CHECK-BE: vcntmbb 1, 2, 1                       # encoding: [0x10,0x39,0x16,0x42]
# CHECK-LE: vcntmbb 1, 2, 1                       # encoding: [0x42,0x16,0x39,0x10]
            vcntmbb 1, 2, 1
# CHECK-BE: vcntmbh 1, 2, 1                       # encoding: [0x10,0x3b,0x16,0x42]
# CHECK-LE: vcntmbh 1, 2, 1                       # encoding: [0x42,0x16,0x3b,0x10]
            vcntmbh 1, 2, 1
# CHECK-BE: vcntmbw 1, 2, 0                       # encoding: [0x10,0x3c,0x16,0x42]
# CHECK-LE: vcntmbw 1, 2, 0                       # encoding: [0x42,0x16,0x3c,0x10]
            vcntmbw 1, 2, 0
# CHECK-BE: vcntmbd 1, 2, 0                       # encoding: [0x10,0x3e,0x16,0x42]
# CHECK-LE: vcntmbd 1, 2, 0                       # encoding: [0x42,0x16,0x3e,0x10]
            vcntmbd 1, 2, 0
# CHECK-BE: vmulesd 1, 2, 3                       # encoding: [0x10,0x22,0x1b,0xc8]
# CHECK-LE: vmulesd 1, 2, 3                       # encoding: [0xc8,0x1b,0x22,0x10]
            vmulesd 1, 2, 3
# CHECK-BE: vmulosd 1, 2, 3                       # encoding: [0x10,0x22,0x19,0xc8]
# CHECK-LE: vmulosd 1, 2, 3                       # encoding: [0xc8,0x19,0x22,0x10]
            vmulosd 1, 2, 3
# CHECK-BE: vmuleud 1, 2, 3                       # encoding: [0x10,0x22,0x1a,0xc8]
# CHECK-LE: vmuleud 1, 2, 3                       # encoding: [0xc8,0x1a,0x22,0x10]
            vmuleud 1, 2, 3
# CHECK-BE: vmuloud 1, 2, 3                       # encoding: [0x10,0x22,0x18,0xc8]
# CHECK-LE: vmuloud 1, 2, 3                       # encoding: [0xc8,0x18,0x22,0x10]
            vmuloud 1, 2, 3
# CHECK-BE: vmsumcud 1, 2, 3, 4                   # encoding: [0x10,0x22,0x19,0x17]
# CHECK-LE: vmsumcud 1, 2, 3, 4                   # encoding: [0x17,0x19,0x22,0x10]
            vmsumcud 1, 2, 3, 4
# CHECK-BE: vdivsq 3, 4, 5                        # encoding: [0x10,0x64,0x29,0x0b]
# CHECK-LE: vdivsq 3, 4, 5                        # encoding: [0x0b,0x29,0x64,0x10]
            vdivsq 3, 4, 5
# CHECK-BE: vdivuq 3, 4, 5                        # encoding: [0x10,0x64,0x28,0x0b]
# CHECK-LE: vdivuq 3, 4, 5                        # encoding: [0x0b,0x28,0x64,0x10]
            vdivuq 3, 4, 5
# CHECK-BE: vdivesq 3, 4, 5                       # encoding: [0x10,0x64,0x2b,0x0b]
# CHECK-LE: vdivesq 3, 4, 5                       # encoding: [0x0b,0x2b,0x64,0x10]
            vdivesq 3, 4, 5
# CHECK-BE: vdiveuq 3, 4, 5                       # encoding: [0x10,0x64,0x2a,0x0b]
# CHECK-LE: vdiveuq 3, 4, 5                       # encoding: [0x0b,0x2a,0x64,0x10]
            vdiveuq 3, 4, 5
# CHECK-BE: vcmpequq 4, 5, 6                      # encoding: [0x10,0x85,0x31,0xc7]
# CHECK-LE: vcmpequq 4, 5, 6                      # encoding: [0xc7,0x31,0x85,0x10]
            vcmpequq 4, 5, 6
# CHECK-BE: vcmpequq. 4, 5, 6                     # encoding: [0x10,0x85,0x35,0xc7]
# CHECK-LE: vcmpequq. 4, 5, 6                     # encoding: [0xc7,0x35,0x85,0x10]
            vcmpequq. 4, 5, 6
# CHECK-BE: vcmpgtsq 4, 5, 6                      # encoding: [0x10,0x85,0x33,0x87]
# CHECK-LE: vcmpgtsq 4, 5, 6                      # encoding: [0x87,0x33,0x85,0x10]
            vcmpgtsq 4, 5, 6
# CHECK-BE: vcmpgtsq. 4, 5, 6                     # encoding: [0x10,0x85,0x37,0x87]
# CHECK-LE: vcmpgtsq. 4, 5, 6                     # encoding: [0x87,0x37,0x85,0x10]
            vcmpgtsq. 4, 5, 6
# CHECK-BE: vcmpgtuq 4, 5, 6                      # encoding: [0x10,0x85,0x32,0x87]
# CHECK-LE: vcmpgtuq 4, 5, 6                      # encoding: [0x87,0x32,0x85,0x10]
            vcmpgtuq 4, 5, 6
# CHECK-BE: vcmpgtuq. 4, 5, 6                     # encoding: [0x10,0x85,0x36,0x87]
# CHECK-LE: vcmpgtuq. 4, 5, 6                     # encoding: [0x87,0x36,0x85,0x10]
            vcmpgtuq. 4, 5, 6
# CHECK-BE: vmoduq 3, 4, 5                        # encoding: [0x10,0x64,0x2e,0x0b]
# CHECK-LE: vmoduq 3, 4, 5                        # encoding: [0x0b,0x2e,0x64,0x10]
            vmoduq 3, 4, 5
# CHECK-BE: vextsd2q 20, 25                       # encoding: [0x12,0x9b,0xce,0x02]
# CHECK-LE: vextsd2q 20, 25                       # encoding: [0x02,0xce,0x9b,0x12]
            vextsd2q 20, 25
# CHECK-BE: vrlq 4, 5, 6                          # encoding: [0x10,0x85,0x30,0x05]
# CHECK-LE: vrlq 4, 5, 6                          # encoding: [0x05,0x30,0x85,0x10]
            vrlq 4, 5, 6
# CHECK-BE: vrlqnm 4, 5, 6                        # encoding: [0x10,0x85,0x31,0x45]
# CHECK-LE: vrlqnm 4, 5, 6                        # encoding: [0x45,0x31,0x85,0x10]
            vrlqnm 4, 5, 6
# CHECK-BE: vrlqmi 4, 5, 6                        # encoding: [0x10,0x85,0x30,0x45]
# CHECK-LE: vrlqmi 4, 5, 6                        # encoding: [0x45,0x30,0x85,0x10]
            vrlqmi 4, 5, 6
# CHECK-BE: vslq 4, 5, 6                          # encoding: [0x10,0x85,0x31,0x05]
# CHECK-LE: vslq 4, 5, 6                          # encoding: [0x05,0x31,0x85,0x10]
            vslq 4, 5, 6
# CHECK-BE: vsrq 4, 5, 6                          # encoding: [0x10,0x85,0x32,0x05]
# CHECK-LE: vsrq 4, 5, 6                          # encoding: [0x05,0x32,0x85,0x10]
            vsrq 4, 5, 6
# CHECK-BE: vsraq 4, 5, 6                         # encoding: [0x10,0x85,0x33,0x05]
# CHECK-LE: vsraq 4, 5, 6                         # encoding: [0x05,0x33,0x85,0x10]
            vsraq 4, 5, 6
# CHECK-BE: xscvqpuqz 8, 28                       # encoding: [0xfd,0x00,0xe6,0x88]
# CHECK-LE: xscvqpuqz 8, 28                       # encoding: [0x88,0xe6,0x00,0xfd]
            xscvqpuqz 8, 28
# CHECK-BE: xscvqpsqz 8, 28                       # encoding: [0xfd,0x08,0xe6,0x88]
# CHECK-LE: xscvqpsqz 8, 28                       # encoding: [0x88,0xe6,0x08,0xfd]
            xscvqpsqz 8, 28
# CHECK-BE: xscvuqqp 8, 28                        # encoding: [0xfd,0x03,0xe6,0x88]
# CHECK-LE: xscvuqqp 8, 28                        # encoding: [0x88,0xe6,0x03,0xfd]
            xscvuqqp 8, 28
# CHECK-BE: xscvsqqp 8, 28                        # encoding: [0xfd,0x0b,0xe6,0x88]
# CHECK-LE: xscvsqqp 8, 28                        # encoding: [0x88,0xe6,0x0b,0xfd]
            xscvsqqp 8, 28
# CHECK-BE: vstribr 2, 2                          # encoding: [0x10,0x41,0x10,0x0d]
# CHECK-LE: vstribr 2, 2                          # encoding: [0x0d,0x10,0x41,0x10]
            vstribr 2, 2
# CHECK-BE: vstribl 2, 2                          # encoding: [0x10,0x40,0x10,0x0d]
# CHECK-LE: vstribl 2, 2                          # encoding: [0x0d,0x10,0x40,0x10]
            vstribl 2, 2
# CHECK-BE: vstrihr 2, 2                          # encoding: [0x10,0x43,0x10,0x0d]
# CHECK-LE: vstrihr 2, 2                          # encoding: [0x0d,0x10,0x43,0x10]
            vstrihr 2, 2
# CHECK-BE: vstribr. 2, 2                         # encoding: [0x10,0x41,0x14,0x0d]
# CHECK-LE: vstribr. 2, 2                         # encoding: [0x0d,0x14,0x41,0x10]
            vstribr. 2, 2
# CHECK-BE: vstribl. 2, 2                         # encoding: [0x10,0x40,0x14,0x0d]
# CHECK-LE: vstribl. 2, 2                         # encoding: [0x0d,0x14,0x40,0x10]
            vstribl. 2, 2
# CHECK-BE: vstrihr. 2, 2                         # encoding: [0x10,0x43,0x14,0x0d]
# CHECK-LE: vstrihr. 2, 2                         # encoding: [0x0d,0x14,0x43,0x10]
            vstrihr. 2, 2
# CHECK-BE: vstrihl. 2, 2                         # encoding: [0x10,0x42,0x14,0x0d]
# CHECK-LE: vstrihl. 2, 2                         # encoding: [0x0d,0x14,0x42,0x10]
            vstrihl. 2, 2
# CHECK-BE: xvcvspbf16 33, 34                     # encoding: [0xf0,0x31,0x17,0x6f]
# CHECK-LE: xvcvspbf16 33, 34                     # encoding: [0x6f,0x17,0x31,0xf0]
            xvcvspbf16 33, 34
# CHECK-BE: xvcvbf16spn 33, 34                    # encoding: [0xf0,0x30,0x17,0x6f]
# CHECK-LE: xvcvbf16spn 33, 34                    # encoding: [0x6f,0x17,0x30,0xf0]
            xvcvbf16spn 33, 34
