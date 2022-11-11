# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-LE %s
# RUN: llvm-mc -triple powerpc-unknown-aix-gnu --show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-BE %s

# CHECK-BE: dmxxextfdmr512 1, 2, 34, 0    # encoding: [0xf0,0x82,0x17,0x12]
# CHECK-LE: dmxxextfdmr512 1, 2, 34, 0    # encoding: [0x12,0x17,0x82,0xf0]
            dmxxextfdmr512 1, 2, 34, 0

# CHECK-BE: dmxxextfdmr512 1, 2, 34, 1    # encoding: [0xf0,0x83,0x17,0x12]
# CHECK-LE: dmxxextfdmr512 1, 2, 34, 1    # encoding: [0x12,0x17,0x83,0xf0]
            dmxxextfdmr512 1, 2, 34, 1

# CHECK-BE: dmxxextfdmr256 3, 8, 0        # encoding: [0xf1,0x80,0x47,0x90]
# CHECK-LE: dmxxextfdmr256 3, 8, 0        # encoding: [0x90,0x47,0x80,0xf1]
            dmxxextfdmr256 3, 8, 0

# CHECK-BE: dmxxextfdmr256 3, 8, 3        # encoding: [0xf1,0x81,0x4f,0x90]
# CHECK-LE: dmxxextfdmr256 3, 8, 3        # encoding: [0x90,0x4f,0x81,0xf1]
            dmxxextfdmr256 3, 8, 3

# CHECK-BE: dmxxinstfdmr512 1, 2, 34, 0   # encoding: [0xf0,0x82,0x17,0x52]
# CHECK-LE: dmxxinstfdmr512 1, 2, 34, 0   # encoding: [0x52,0x17,0x82,0xf0]
            dmxxinstfdmr512 1, 2, 34, 0

# CHECK-BE: dmxxinstfdmr512 1, 2, 34, 1   # encoding: [0xf0,0x83,0x17,0x52]
# CHECK-LE: dmxxinstfdmr512 1, 2, 34, 1   # encoding: [0x52,0x17,0x83,0xf0]
            dmxxinstfdmr512 1, 2, 34, 1

# CHECK-BE: dmxxinstfdmr256 3, 8, 0       # encoding: [0xf1,0x80,0x47,0x94]
# CHECK-LE: dmxxinstfdmr256 3, 8, 0       # encoding: [0x94,0x47,0x80,0xf1]
            dmxxinstfdmr256 3, 8, 0

# CHECK-BE: dmxxinstfdmr256 3, 8, 3       # encoding: [0xf1,0x81,0x4f,0x94]
# CHECK-LE: dmxxinstfdmr256 3, 8, 3       # encoding: [0x94,0x4f,0x81,0xf1]
            dmxxinstfdmr256 3, 8, 3

# CHECK-BE: dmsetdmrz 3                   # encoding: [0x7d,0x82,0x01,0x62]
# CHECK-LE: dmsetdmrz 3                   # encoding: [0x62,0x01,0x82,0x7d]
            dmsetdmrz 3

# CHECK-BE: dmmr 4, 5                     # encoding: [0x7e,0x06,0xa1,0x62]
# CHECK-LE: dmmr 4, 5                     # encoding: [0x62,0xa1,0x06,0x7e]
            dmmr 4, 5

# CHECK-BE: dmxor 6, 7                    # encoding: [0x7f,0x07,0xe1,0x62]
# CHECK-LE: dmxor 6, 7                    # encoding: [0x62,0xe1,0x07,0x7f]
            dmxor 6, 7

# CHECK-BE: subfus 3, 0, 4, 5          # encoding: [0x7c,0x64,0x28,0x90]
# CHECK-LE: subfus 3, 0, 4, 5          # encoding: [0x90,0x28,0x64,0x7c]
            subfus 3, 0, 4, 5

# CHECK-BE: subfus 3, 1, 4, 5          # encoding: [0x7c,0x64,0x2c,0x90]
# CHECK-LE: subfus 3, 1, 4, 5          # encoding: [0x90,0x2c,0x64,0x7c]
            subfus 3, 1, 4, 5

# CHECK-BE: subfus. 3, 0, 4, 5         # encoding: [0x7c,0x64,0x28,0x91]
# CHECK-LE: subfus. 3, 0, 4, 5         # encoding: [0x91,0x28,0x64,0x7c]
            subfus. 3, 0, 4, 5

# CHECK-BE: subfus. 3, 1, 4, 5         # encoding: [0x7c,0x64,0x2c,0x91]
# CHECK-LE: subfus. 3, 1, 4, 5         # encoding: [0x91,0x2c,0x64,0x7c]
            subfus. 3, 1, 4, 5
