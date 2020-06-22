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
# CHECK-BE: vclrlb 1, 4, 3                        # encoding: [0x10,0x24,0x19,0x8d]
# CHECK-LE: vclrlb 1, 4, 3                        # encoding: [0x8d,0x19,0x24,0x10]
            vclrlb 1, 4, 3
# CHECK-BE: vclrrb 1, 4, 3                        # encoding: [0x10,0x24,0x19,0xcd]
# CHECK-LE: vclrrb 1, 4, 3                        # encoding: [0xcd,0x19,0x24,0x10]
            vclrrb 1, 4, 3
