# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Instructions specific to the PowerPC 4xx embedded controllers:

# CHECK-BE: mfdcr 3, 178                     # encoding: [0x7c,0x72,0x2a,0x86]
# CHECK-LE: mfdcr 3, 178                     # encoding: [0x86,0x2a,0x72,0x7c]
            mfdcr 3,178
# CHECK-BE: mtdcr 178, 3                     # encoding: [0x7c,0x72,0x2b,0x86]
# CHECK-LE: mtdcr 178, 3                     # encoding: [0x86,0x2b,0x72,0x7c]
            mtdcr 178,3

# CHECK-BE: tlbrehi 2, 3                     # encoding: [0x7c,0x43,0x07,0x64]
# CHECK-LE: tlbrehi 2, 3                     # encoding: [0x64,0x07,0x43,0x7c]
            tlbre %r2, %r3, 0
# CHECK-BE: tlbrelo 2, 3                     # encoding: [0x7c,0x43,0x0f,0x64]
# CHECK-LE: tlbrelo 2, 3                     # encoding: [0x64,0x0f,0x43,0x7c]
            tlbre %r2, %r3, 1
# CHECK-BE: tlbrehi 2, 3                     # encoding: [0x7c,0x43,0x07,0x64]
# CHECK-LE: tlbrehi 2, 3                     # encoding: [0x64,0x07,0x43,0x7c]
            tlbrehi %r2, %r3
# CHECK-BE: tlbrelo 2, 3                     # encoding: [0x7c,0x43,0x0f,0x64]
# CHECK-LE: tlbrelo 2, 3                     # encoding: [0x64,0x0f,0x43,0x7c]
            tlbrelo %r2, %r3

# CHECK-BE: tlbwehi 2, 3                     # encoding: [0x7c,0x43,0x07,0xa4]
# CHECK-LE: tlbwehi 2, 3                     # encoding: [0xa4,0x07,0x43,0x7c]
            tlbwe %r2, %r3, 0
# CHECK-BE: tlbwelo 2, 3                     # encoding: [0x7c,0x43,0x0f,0xa4]
# CHECK-LE: tlbwelo 2, 3                     # encoding: [0xa4,0x0f,0x43,0x7c]
            tlbwe %r2, %r3, 1
# CHECK-BE: tlbwehi 2, 3                     # encoding: [0x7c,0x43,0x07,0xa4]
# CHECK-LE: tlbwehi 2, 3                     # encoding: [0xa4,0x07,0x43,0x7c]
            tlbwehi %r2, %r3
# CHECK-BE: tlbwelo 2, 3                     # encoding: [0x7c,0x43,0x0f,0xa4]
# CHECK-LE: tlbwelo 2, 3                     # encoding: [0xa4,0x0f,0x43,0x7c]
            tlbwelo %r2, %r3

# CHECK-BE: tlbsx 2, 3, 1                    # encoding: [0x7c,0x43,0x0f,0x24]
# CHECK-LE: tlbsx 2, 3, 1                    # encoding: [0x24,0x0f,0x43,0x7c]
            tlbsx %r2, %r3, %r1
# CHECK-BE: tlbsx. 2, 3, 1                   # encoding: [0x7c,0x43,0x0f,0x25]
# CHECK-LE: tlbsx. 2, 3, 1                   # encoding: [0x25,0x0f,0x43,0x7c]
            tlbsx. %r2, %r3, %r1

# CHECK-BE: mfdccr 2                         # encoding: [0x7c,0x5a,0xfa,0xa6]
# CHECK-LE: mfdccr 2                         # encoding: [0xa6,0xfa,0x5a,0x7c]
            mfdccr %r2
# CHECK-BE: mtdccr 2                         # encoding: [0x7c,0x5a,0xfb,0xa6]
# CHECK-LE: mtdccr 2                         # encoding: [0xa6,0xfb,0x5a,0x7c]
            mtdccr %r2

# CHECK-BE: mficcr 2                         # encoding: [0x7c,0x5b,0xfa,0xa6]
# CHECK-LE: mficcr 2                         # encoding: [0xa6,0xfa,0x5b,0x7c]
            mficcr %r2
# CHECK-BE: mticcr 2                         # encoding: [0x7c,0x5b,0xfb,0xa6]
# CHECK-LE: mticcr 2                         # encoding: [0xa6,0xfb,0x5b,0x7c]
            mticcr %r2

# CHECK-BE: mfdear 2                        # encoding: [0x7c,0x55,0xf2,0xa6]
# CHECK-LE: mfdear 2                        # encoding: [0xa6,0xf2,0x55,0x7c]
            mfdear %r2
# CHECK-BE: mtdear 2                        # encoding: [0x7c,0x55,0xf3,0xa6]
# CHECK-LE: mtdear 2                        # encoding: [0xa6,0xf3,0x55,0x7c]
            mtdear %r2

# CHECK-BE: mfesr 2                         # encoding: [0x7c,0x54,0xf2,0xa6]
# CHECK-LE: mfesr 2                         # encoding: [0xa6,0xf2,0x54,0x7c]
            mfesr %r2
# CHECK-BE: mtesr 2                         # encoding: [0x7c,0x54,0xf3,0xa6]
# CHECK-LE: mtesr 2                         # encoding: [0xa6,0xf3,0x54,0x7c]
            mtesr %r2

# CHECK-BE: mftcr 2                         # encoding: [0x7c,0x5a,0xf2,0xa6]
# CHECK-LE: mftcr 2                         # encoding: [0xa6,0xf2,0x5a,0x7c]
            mftcr %r2
# CHECK-BE: mttcr 2                         # encoding: [0x7c,0x5a,0xf3,0xa6]
# CHECK-LE: mttcr 2                         # encoding: [0xa6,0xf3,0x5a,0x7c]
            mttcr %r2

# CHECK-BE: mftblo 2                        # encoding: [0x7c,0x5d,0xf2,0xa6]
# CHECK-LE: mftblo 2                        # encoding: [0xa6,0xf2,0x5d,0x7c]
            mftblo %r2
# CHECK-BE: mttblo 2                        # encoding: [0x7c,0x5d,0xf3,0xa6]
# CHECK-LE: mttblo 2                        # encoding: [0xa6,0xf3,0x5d,0x7c]
            mttblo %r2
# CHECK-BE: mftbhi 2                        # encoding: [0x7c,0x5c,0xf2,0xa6]
# CHECK-LE: mftbhi 2                        # encoding: [0xa6,0xf2,0x5c,0x7c]
            mftbhi %r2
# CHECK-BE: mttbhi 2                        # encoding: [0x7c,0x5c,0xf3,0xa6]
# CHECK-LE: mttbhi 2                        # encoding: [0xa6,0xf3,0x5c,0x7c]
            mttbhi %r2

# CHECK-BE: dccci 5, 6                      # encoding: [0x7c,0x05,0x33,0x8c]
# CHECK-LE: dccci 5, 6                      # encoding: [0x8c,0x33,0x05,0x7c]
            dccci %r5,%r6
# CHECK-BE: iccci 5, 6                      # encoding: [0x7c,0x05,0x37,0x8c]
# CHECK-LE: iccci 5, 6                      # encoding: [0x8c,0x37,0x05,0x7c]
            iccci %r5,%r6
# CHECK-BE: dccci 0, 0                      # encoding: [0x7c,0x00,0x03,0x8c]
# CHECK-LE: dccci 0, 0                      # encoding: [0x8c,0x03,0x00,0x7c]
            dci %r0
# CHECK-BE: iccci 0, 0                      # encoding: [0x7c,0x00,0x07,0x8c]
# CHECK-LE: iccci 0, 0                      # encoding: [0x8c,0x07,0x00,0x7c]
            ici 0

# CHECK-BE: mfsrr2 2                        # encoding: [0x7c,0x5e,0xf2,0xa6]
# CHECK-LE: mfsrr2 2                        # encoding: [0xa6,0xf2,0x5e,0x7c]
            mfsrr2 2
# CHECK-BE: mtsrr2 2                        # encoding: [0x7c,0x5e,0xf3,0xa6]
# CHECK-LE: mtsrr2 2                        # encoding: [0xa6,0xf3,0x5e,0x7c]
            mtsrr2 2
# CHECK-BE: mfsrr3 2                        # encoding: [0x7c,0x5f,0xf2,0xa6]
# CHECK-LE: mfsrr3 2                        # encoding: [0xa6,0xf2,0x5f,0x7c]
            mfsrr3 2
# CHECK-BE: mtsrr3 2                        # encoding: [0x7c,0x5f,0xf3,0xa6]
# CHECK-LE: mtsrr3 2                        # encoding: [0xa6,0xf3,0x5f,0x7c]
            mtsrr3 2

# CHECK-BE: mfbr0 5                         # encoding: [0x7c,0xa0,0x22,0x86]
# CHECK-LE: mfbr0 5                         # encoding: [0x86,0x22,0xa0,0x7c]
            mfbr0 %r5
# CHECK-BE: mtbr0 5                         # encoding: [0x7c,0xa0,0x23,0x86]
# CHECK-LE: mtbr0 5                         # encoding: [0x86,0x23,0xa0,0x7c]
            mtbr0 %r5
# CHECK-BE: mfbr1 5                         # encoding: [0x7c,0xa1,0x22,0x86]
# CHECK-LE: mfbr1 5                         # encoding: [0x86,0x22,0xa1,0x7c]
            mfbr1 %r5
# CHECK-BE: mtbr1 5                         # encoding: [0x7c,0xa1,0x23,0x86]
# CHECK-LE: mtbr1 5                         # encoding: [0x86,0x23,0xa1,0x7c]
            mtbr1 %r5
# CHECK-BE: mfbr2 5                         # encoding: [0x7c,0xa2,0x22,0x86]
# CHECK-LE: mfbr2 5                         # encoding: [0x86,0x22,0xa2,0x7c]
            mfbr2 %r5
# CHECK-BE: mtbr2 5                         # encoding: [0x7c,0xa2,0x23,0x86]
# CHECK-LE: mtbr2 5                         # encoding: [0x86,0x23,0xa2,0x7c]
            mtbr2 %r5
# CHECK-BE: mfbr3 5                         # encoding: [0x7c,0xa3,0x22,0x86]
# CHECK-LE: mfbr3 5                         # encoding: [0x86,0x22,0xa3,0x7c]
            mfbr3 %r5
# CHECK-BE: mtbr3 5                         # encoding: [0x7c,0xa3,0x23,0x86]
# CHECK-LE: mtbr3 5                         # encoding: [0x86,0x23,0xa3,0x7c]
            mtbr3 %r5
# CHECK-BE: mfbr4 5                         # encoding: [0x7c,0xa4,0x22,0x86]
# CHECK-LE: mfbr4 5                         # encoding: [0x86,0x22,0xa4,0x7c]
            mfbr4 %r5
# CHECK-BE: mtbr4 5                         # encoding: [0x7c,0xa4,0x23,0x86]
# CHECK-LE: mtbr4 5                         # encoding: [0x86,0x23,0xa4,0x7c]
            mtbr4 %r5
# CHECK-BE: mfbr5 5                         # encoding: [0x7c,0xa5,0x22,0x86]
# CHECK-LE: mfbr5 5                         # encoding: [0x86,0x22,0xa5,0x7c]
            mfbr5 %r5
# CHECK-BE: mtbr5 5                         # encoding: [0x7c,0xa5,0x23,0x86]
# CHECK-LE: mtbr5 5                         # encoding: [0x86,0x23,0xa5,0x7c]
            mtbr5 %r5
# CHECK-BE: mfbr6 5                         # encoding: [0x7c,0xa6,0x22,0x86]
# CHECK-LE: mfbr6 5                         # encoding: [0x86,0x22,0xa6,0x7c]
            mfbr6 %r5
# CHECK-BE: mtbr6 5                         # encoding: [0x7c,0xa6,0x23,0x86]
# CHECK-LE: mtbr6 5                         # encoding: [0x86,0x23,0xa6,0x7c]
            mtbr6 %r5
# CHECK-BE: mfbr7 5                         # encoding: [0x7c,0xa7,0x22,0x86]
# CHECK-LE: mfbr7 5                         # encoding: [0x86,0x22,0xa7,0x7c]
            mfbr7 %r5
# CHECK-BE: mtbr7 5                         # encoding: [0x7c,0xa7,0x23,0x86]
# CHECK-LE: mtbr7 5                         # encoding: [0x86,0x23,0xa7,0x7c]
            mtbr7 %r5
