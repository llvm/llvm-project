# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding -ppc-asm-full-reg-names %s | \
# RUN:   FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --show-encoding -ppc-asm-full-reg-names %s | \
# RUN:   FileCheck -check-prefix=CHECK-LE %s

# CHECK-BE: addi r1, r2, 700                   # encoding: [0x38,0x22,0x02,0xbc]
# CHECK-LE: addi r1, r2, 700                   # encoding: [0xbc,0x02,0x22,0x38]
            addi 1, 2, 700

# CHECK-BE: li r1, 700                         # encoding: [0x38,0x20,0x02,0xbc]
# CHECK-LE: li r1, 700                         # encoding: [0xbc,0x02,0x20,0x38]
            addi 1, 0, 700

# CHECK-BE: paddi r1, r2, 6400000, 0           # encoding: [0x06,0x00,0x00,0x61,
# CHECK-BE-SAME:                                            0x38,0x22,0xa8,0x00]
# CHECK-LE: paddi r1, r2, 6400000, 0           # encoding: [0x61,0x00,0x00,0x06,
# CHECK-LE-SAME:                                            0x00,0xa8,0x22,0x38]
            paddi 1, 2, 6400000, 0

# CHECK-BE: paddi r1, 0, 6400000, 0            # encoding: [0x06,0x00,0x00,0x61,
# CHECK-BE-SAME:                                            0x38,0x20,0xa8,0x00]
# CHECK-LE: paddi r1, 0, 6400000, 0            # encoding: [0x61,0x00,0x00,0x06,
# CHECK-LE-SAME:                                            0x00,0xa8,0x20,0x38]
            paddi 1, 0, 6400000, 0
