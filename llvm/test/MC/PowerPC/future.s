# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-LE %s

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

