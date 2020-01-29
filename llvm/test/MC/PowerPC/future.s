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

