# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-LE %s

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
