# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# CHECK-BE: hrfid                            # encoding: [0x4c,0x00,0x02,0x24]
# CHECK-LE: hrfid                            # encoding: [0x24,0x02,0x00,0x4c]
            hrfid

# CHECK-BE: nap                              # encoding: [0x4c,0x00,0x03,0x64]
# CHECK-LE: nap                              # encoding: [0x64,0x03,0x00,0x4c]
            nap   

# CHECK-BE: mtmsr 4                          # encoding: [0x7c,0x80,0x01,0x24]
# CHECK-LE: mtmsr 4                          # encoding: [0x24,0x01,0x80,0x7c]
            mtmsr %r4

# CHECK-BE: mtmsr 4, 1                       # encoding: [0x7c,0x81,0x01,0x24]
# CHECK-LE: mtmsr 4, 1                       # encoding: [0x24,0x01,0x81,0x7c]
            mtmsr %r4, 1

# CHECK-BE: mfmsr 4                         # encoding: [0x7c,0x80,0x00,0xa6]
# CHECK-LE: mfmsr 4                         # encoding: [0xa6,0x00,0x80,0x7c]
            mfmsr %r4

# CHECK-BE: mtmsrd 4                        # encoding: [0x7c,0x80,0x01,0x64]
# CHECK-LE: mtmsrd 4                        # encoding: [0x64,0x01,0x80,0x7c]
            mtmsrd %r4

# CHECK-BE: mtmsrd 4, 1                     # encoding: [0x7c,0x81,0x01,0x64]
# CHECK-LE: mtmsrd 4, 1                     # encoding: [0x64,0x01,0x81,0x7c]
            mtmsrd %r4, 1

# CHECK-BE: mfspr 4, 260                    # encoding: [0x7c,0x84,0x42,0xa6]
# CHECK-LE: mfspr 4, 260                    # encoding: [0xa6,0x42,0x84,0x7c]
            mfsprg %r4, 4

# CHECK-BE: mfspr 4, 261                    # encoding: [0x7c,0x85,0x42,0xa6]
# CHECK-LE: mfspr 4, 261                    # encoding: [0xa6,0x42,0x85,0x7c]
            mfsprg %r4, 5

# CHECK-BE: mfspr 4, 262                    # encoding: [0x7c,0x86,0x42,0xa6]
# CHECK-LE: mfspr 4, 262                    # encoding: [0xa6,0x42,0x86,0x7c]
            mfsprg %r4, 6

# CHECK-BE: mfspr 4, 263                    # encoding: [0x7c,0x87,0x42,0xa6]
# CHECK-LE: mfspr 4, 263                    # encoding: [0xa6,0x42,0x87,0x7c]
            mfsprg %r4, 7

# CHECK-BE: mfspr 2, 260                    # encoding: [0x7c,0x44,0x42,0xa6]
# CHECK-LE: mfspr 2, 260                    # encoding: [0xa6,0x42,0x44,0x7c]
            mfsprg4 %r2
# CHECK-BE: mfspr 2, 261                    # encoding: [0x7c,0x45,0x42,0xa6]
# CHECK-LE: mfspr 2, 261                    # encoding: [0xa6,0x42,0x45,0x7c]
            mfsprg5 %r2
# CHECK-BE: mfspr 2, 262                    # encoding: [0x7c,0x46,0x42,0xa6]
# CHECK-LE: mfspr 2, 262                    # encoding: [0xa6,0x42,0x46,0x7c]
            mfsprg6 %r2
# CHECK-BE: mfspr 2, 263                    # encoding: [0x7c,0x47,0x42,0xa6]
# CHECK-LE: mfspr 2, 263                    # encoding: [0xa6,0x42,0x47,0x7c]
            mfsprg7 %r2

# NOT-CHECK-BE: mtspr 260, 4                    # encoding: [0x7c,0x90,0x43,0xa6]
# NOT-CHECK-LE: mtspr 260, 4                    # encoding: [0xa6,0x43,0x90,0x7c]
            mtsprg 4, %r4

# NOT-CHECK-BE: mtspr 261, 4                    # encoding: [0x7c,0x91,0x43,0xa6]
# NOT-CHECK-LE: mtspr 261, 4                    # encoding: [0xa6,0x43,0x91,0x7c]
            mtsprg 5, %r4

# NOT-CHECK-BE: mtspr 262, 4                    # encoding: [0x7c,0x92,0x43,0xa6]
# NOT-CHECK-LE: mtspr 262, 4                    # encoding: [0xa6,0x43,0x92,0x7c]
            mtsprg 6, %r4

# NOT-CHECK-BE: mtspr 263, 4                    # encoding: [0x7c,0x93,0x43,0xa6]
# NOT-CHECK-LE: mtspr 263, 4                    # encoding: [0xa6,0x43,0x93,0x7c]
            mtsprg 7, %r4

# CHECK-BE: mtspr 260, 4                    # encoding: [0x7c,0x84,0x43,0xa6]
# CHECK-LE: mtspr 260, 4                    # encoding: [0xa6,0x43,0x84,0x7c]
            mtsprg4 %r4

# CHECK-BE: mtspr 261, 4                    # encoding: [0x7c,0x85,0x43,0xa6]
# CHECK-LE: mtspr 261, 4                    # encoding: [0xa6,0x43,0x85,0x7c]
            mtsprg5 %r4

# CHECK-BE: mtspr 262, 4                    # encoding: [0x7c,0x86,0x43,0xa6]
# CHECK-LE: mtspr 262, 4                    # encoding: [0xa6,0x43,0x86,0x7c]
            mtsprg6 %r4

# CHECK-BE: mtspr 263, 4                    # encoding: [0x7c,0x87,0x43,0xa6]
# CHECK-LE: mtspr 263, 4                    # encoding: [0xa6,0x43,0x87,0x7c]
            mtsprg7 %r4

# CHECK-BE: mtspr 280, 4                    # encoding: [0x7c,0x98,0x43,0xa6]
# CHECK-LE: mtspr 280, 4                    # encoding: [0xa6,0x43,0x98,0x7c]
            mtasr %r4

# CHECK-BE: mfspr 4, 22                     # encoding: [0x7c,0x96,0x02,0xa6]
# CHECK-LE: mfspr 4, 22                     # encoding: [0xa6,0x02,0x96,0x7c]
            mfdec %r4

# CHECK-BE: mtspr 22, 4                     # encoding: [0x7c,0x96,0x03,0xa6]
# CHECK-LE: mtspr 22, 4                     # encoding: [0xa6,0x03,0x96,0x7c]
            mtdec %r4

# CHECK-BE: mfpvr 4                         # encoding: [0x7c,0x9f,0x42,0xa6]
# CHECK-LE: mfpvr 4                         # encoding: [0xa6,0x42,0x9f,0x7c]
            mfpvr %r4

# CHECK-BE: mfspr 4, 25                     # encoding: [0x7c,0x99,0x02,0xa6]
# CHECK-LE: mfspr 4, 25                     # encoding: [0xa6,0x02,0x99,0x7c]
            mfsdr1 %r4

# CHECK-BE: mtspr 25, 4                     # encoding: [0x7c,0x99,0x03,0xa6]
# CHECK-LE: mtspr 25, 4                     # encoding: [0xa6,0x03,0x99,0x7c]
            mtsdr1 %r4

# CHECK-BE: mfspr 4, 26                     # encoding: [0x7c,0x9a,0x02,0xa6]
# CHECK-LE: mfspr 4, 26                     # encoding: [0xa6,0x02,0x9a,0x7c]
            mfsrr0 %r4

# CHECK-BE: mtspr 26, 4                     # encoding: [0x7c,0x9a,0x03,0xa6]
# CHECK-LE: mtspr 26, 4                     # encoding: [0xa6,0x03,0x9a,0x7c]
            mtsrr0 %r4

# CHECK-BE: mfspr 4, 27                     # encoding: [0x7c,0x9b,0x02,0xa6]
# CHECK-LE: mfspr 4, 27                     # encoding: [0xa6,0x02,0x9b,0x7c]
            mfsrr1 %r4

# CHECK-BE: mtspr 27, 4                     # encoding: [0x7c,0x9b,0x03,0xa6]
# CHECK-LE: mtspr 27, 4                     # encoding: [0xa6,0x03,0x9b,0x7c]
            mtsrr1 %r4

# CHECK-BE: slbie 4                         # encoding: [0x7c,0x00,0x23,0x64]
# CHECK-LE: slbie 4                         # encoding: [0x64,0x23,0x00,0x7c]
            slbie %r4

# CHECK-BE: slbmte 4, 5                     # encoding: [0x7c,0x80,0x2b,0x24]
# CHECK-LE: slbmte 4, 5                     # encoding: [0x24,0x2b,0x80,0x7c]
            slbmte %r4, %r5

# CHECK-BE: slbmfee 4, 5                    # encoding: [0x7c,0x80,0x2f,0x26]
# CHECK-LE: slbmfee 4, 5                    # encoding: [0x26,0x2f,0x80,0x7c]
            slbmfee %r4, %r5

# CHECK-BE: slbmfev 2, 3                    # encoding: [0x7c,0x40,0x1e,0xa6]
# CHECK-LE: slbmfev 2, 3                    # encoding: [0xa6,0x1e,0x40,0x7c]
            slbmfev %r2, %r3

# CHECK-BE: slbia                           # encoding: [0x7c,0x00,0x03,0xe4]
# CHECK-LE: slbia                           # encoding: [0xe4,0x03,0x00,0x7c]
            slbia

# CHECK-BE: tlbsync                         # encoding: [0x7c,0x00,0x04,0x6c]
# CHECK-LE: tlbsync                         # encoding: [0x6c,0x04,0x00,0x7c]
            tlbsync

# CHECK-BE: tlbiel 4                        # encoding: [0x7c,0x00,0x22,0x24]
# CHECK-LE: tlbiel 4                        # encoding: [0x24,0x22,0x00,0x7c]
            tlbiel %r4

# CHECK-BE: tlbie 4                         # encoding: [0x7c,0x00,0x22,0x64]
# CHECK-LE: tlbie 4                         # encoding: [0x64,0x22,0x00,0x7c]
            tlbie %r4, 0

# CHECK-BE: tlbie 4                         # encoding: [0x7c,0x00,0x22,0x64]
# CHECK-LE: tlbie 4                         # encoding: [0x64,0x22,0x00,0x7c]
            tlbie %r4

# CHECK-BE: rfi                             # encoding: [0x4c,0x00,0x00,0x64]
# CHECK-LE: rfi                             # encoding: [0x64,0x00,0x00,0x4c]
            rfi
# CHECK-BE: rfci                            # encoding: [0x4c,0x00,0x00,0x66]
# CHECK-LE: rfci                            # encoding: [0x66,0x00,0x00,0x4c]
            rfci

# CHECK-BE: wrtee 12                        # encoding: [0x7d,0x80,0x01,0x06]
# CHECK-LE: wrtee 12                        # encoding: [0x06,0x01,0x80,0x7d]
            wrtee %r12

# CHECK-BE: wrteei 0                        # encoding: [0x7c,0x00,0x01,0x46]
# CHECK-LE: wrteei 0                        # encoding: [0x46,0x01,0x00,0x7c]
            wrteei 0

# CHECK-BE: wrteei 1                        # encoding: [0x7c,0x00,0x81,0x46]
# CHECK-LE: wrteei 1                        # encoding: [0x46,0x81,0x00,0x7c]
            wrteei 1

# CHECK-BE: tlbre                           # encoding: [0x7c,0x00,0x07,0x64]
# CHECK-LE: tlbre                           # encoding: [0x64,0x07,0x00,0x7c]
            tlbre
# CHECK-BE: tlbwe                           # encoding: [0x7c,0x00,0x07,0xa4]
# CHECK-LE: tlbwe                           # encoding: [0xa4,0x07,0x00,0x7c]
            tlbwe
# CHECK-BE: tlbivax 11, 12                  # encoding: [0x7c,0x0b,0x66,0x24]
# CHECK-LE: tlbivax 11, 12                  # encoding: [0x24,0x66,0x0b,0x7c]
            tlbivax %r11, %r12
# CHECK-BE: tlbsx 11, 12                    # encoding: [0x7c,0x0b,0x67,0x24]
# CHECK-LE: tlbsx 11, 12                    # encoding: [0x24,0x67,0x0b,0x7c]
            tlbsx %r11, %r12

# CHECK-BE: mfpmr 5, 400                    # encoding: [0x7c,0xb0,0x62,0x9c]
# CHECK-LE: mfpmr 5, 400                    # encoding: [0x9c,0x62,0xb0,0x7c]
            mfpmr 5, 400
# CHECK-BE: mtpmr 400, 6                    # encoding: [0x7c,0xd0,0x63,0x9c]
# CHECK-LE: mtpmr 400, 6                    # encoding: [0x9c,0x63,0xd0,0x7c]
            mtpmr 400, 6
# CHECK-BE: icblc 0, 0, 8                      # encoding: [0x7c,0x00,0x41,0xcc]
# CHECK-LE: icblc 0, 0, 8                      # encoding: [0xcc,0x41,0x00,0x7c]
            icblc 0, 0, 8
# CHECK-BE: icbtls 0, 0, 9                     # encoding: [0x7c,0x00,0x4b,0xcc]
# CHECK-LE: icbtls 0, 0, 9                     # encoding: [0xcc,0x4b,0x00,0x7c]
            icbtls 0, 0, 9
