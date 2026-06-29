# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu -mcpu=pwr8 -show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu -mcpu=pwr8 -show-encoding %s | \
# RUN:   FileCheck -check-prefix=CHECK-LE %s

# ISA 2.07 (POWER8) Branch Conditional to Target Address Register

# bctar (generic form, 3-operand)
# CHECK-BE: bctar 4, 10, 3                  # encoding: [0x4c,0x8a,0x1c,0x60]
# CHECK-LE: bctar 4, 10, 3                  # encoding: [0x60,0x1c,0x8a,0x4c]
            bctar 4, 10, 3
# CHECK-BE: bctar 4, 10                     # encoding: [0x4c,0x8a,0x04,0x60]
# CHECK-LE: bctar 4, 10                     # encoding: [0x60,0x04,0x8a,0x4c]
            bctar 4, 10
# CHECK-BE: bctarl 4, 10, 3                 # encoding: [0x4c,0x8a,0x1c,0x61]
# CHECK-LE: bctarl 4, 10, 3                 # encoding: [0x61,0x1c,0x8a,0x4c]
            bctarl 4, 10, 3
# CHECK-BE: bctarl 4, 10                    # encoding: [0x4c,0x8a,0x04,0x61]
# CHECK-LE: bctarl 4, 10                    # encoding: [0x61,0x04,0x8a,0x4c]
            bctarl 4, 10

# bttar/bftar (simple mnemonics print as generic bctar)
# CHECK-BE: bctar 12, 2                     # encoding: [0x4d,0x82,0x04,0x60]
# CHECK-LE: bctar 12, 2                     # encoding: [0x60,0x04,0x82,0x4d]
            bttar 2
# CHECK-BE: bctar 4, 2                      # encoding: [0x4c,0x82,0x04,0x60]
# CHECK-LE: bctar 4, 2                      # encoding: [0x60,0x04,0x82,0x4c]
            bftar 2
# CHECK-BE: bctarl 12, 2                    # encoding: [0x4d,0x82,0x04,0x61]
# CHECK-LE: bctarl 12, 2                    # encoding: [0x61,0x04,0x82,0x4d]
            bttarl 2
# CHECK-BE: bctarl 4, 2                     # encoding: [0x4c,0x82,0x04,0x61]
# CHECK-LE: bctarl 4, 2                     # encoding: [0x61,0x04,0x82,0x4c]
            bftarl 2

# bttar/bftar with prediction hints
# CHECK-BE: bctar 15, 2                     # encoding: [0x4d,0xe2,0x04,0x60]
# CHECK-LE: bctar 15, 2                     # encoding: [0x60,0x04,0xe2,0x4d]
            bttar+ 2
# CHECK-BE: bctar 14, 2                     # encoding: [0x4d,0xc2,0x04,0x60]
# CHECK-LE: bctar 14, 2                     # encoding: [0x60,0x04,0xc2,0x4d]
            bttar- 2
# CHECK-BE: bctar 7, 2                      # encoding: [0x4c,0xe2,0x04,0x60]
# CHECK-LE: bctar 7, 2                      # encoding: [0x60,0x04,0xe2,0x4c]
            bftar+ 2
# CHECK-BE: bctar 6, 2                      # encoding: [0x4c,0xc2,0x04,0x60]
# CHECK-LE: bctar 6, 2                      # encoding: [0x60,0x04,0xc2,0x4c]
            bftar- 2

# Extended mnemonics (blttar, bgttar, etc.)
# CHECK-BE: blttar 0                        # encoding: [0x4d,0x80,0x04,0x60]
# CHECK-LE: blttar 0                        # encoding: [0x60,0x04,0x80,0x4d]
            blttar
# CHECK-BE: blttar 2                        # encoding: [0x4d,0x88,0x04,0x60]
# CHECK-LE: blttar 2                        # encoding: [0x60,0x04,0x88,0x4d]
            blttar 2
# CHECK-BE: blttarl 0                       # encoding: [0x4d,0x80,0x04,0x61]
# CHECK-LE: blttarl 0                       # encoding: [0x61,0x04,0x80,0x4d]
            blttarl
# CHECK-BE: blttarl 2                       # encoding: [0x4d,0x88,0x04,0x61]
# CHECK-LE: blttarl 2                       # encoding: [0x61,0x04,0x88,0x4d]
            blttarl 2

# CHECK-BE: bgttar 2                        # encoding: [0x4d,0x89,0x04,0x60]
# CHECK-LE: bgttar 2                        # encoding: [0x60,0x04,0x89,0x4d]
            bgttar 2
# CHECK-BE: beqtar 2                        # encoding: [0x4d,0x8a,0x04,0x60]
# CHECK-LE: beqtar 2                        # encoding: [0x60,0x04,0x8a,0x4d]
            beqtar 2

# Extended mnemonics with prediction hints
# CHECK-BE: blttar+ 0                       # encoding: [0x4d,0xe0,0x04,0x60]
# CHECK-LE: blttar+ 0                       # encoding: [0x60,0x04,0xe0,0x4d]
            blttar+
# CHECK-BE: blttar- 0                       # encoding: [0x4d,0xc0,0x04,0x60]
# CHECK-LE: blttar- 0                       # encoding: [0x60,0x04,0xc0,0x4d]
            blttar-
# CHECK-BE: blttarl+ 0                      # encoding: [0x4d,0xe0,0x04,0x61]
# CHECK-LE: blttarl+ 0                      # encoding: [0x61,0x04,0xe0,0x4d]
            blttarl+
# CHECK-BE: blttarl- 0                      # encoding: [0x4d,0xc0,0x04,0x61]
# CHECK-LE: blttarl- 0                      # encoding: [0x61,0x04,0xc0,0x4d]
            blttarl-

# bdnztar/bdztar
# CHECK-BE: bctar 16, 0                  # encoding: [0x4e,0x00,0x04,0x60]
# CHECK-LE: bctar 16, 0                  # encoding: [0x60,0x04,0x00,0x4e]
            bdnztar
# CHECK-BE: bctar 18, 0                  # encoding: [0x4e,0x40,0x04,0x60]
# CHECK-LE: bctar 18, 0                  # encoding: [0x60,0x04,0x40,0x4e]
            bdztar
# CHECK-BE: bctarl 16, 0                 # encoding: [0x4e,0x00,0x04,0x61]
# CHECK-LE: bctarl 16, 0                 # encoding: [0x61,0x04,0x00,0x4e]
            bdnztarl
# CHECK-BE: bctarl 18, 0                 # encoding: [0x4e,0x40,0x04,0x61]
# CHECK-LE: bctarl 18, 0                 # encoding: [0x61,0x04,0x40,0x4e]
            bdztarl

# bdnztar/bdztar with prediction hints
# CHECK-BE: bctar 25, 0                  # encoding: [0x4f,0x20,0x04,0x60]
# CHECK-LE: bctar 25, 0                  # encoding: [0x60,0x04,0x20,0x4f]
            bdnztar+
# CHECK-BE: bctar 24, 0                  # encoding: [0x4f,0x00,0x04,0x60]
# CHECK-LE: bctar 24, 0                  # encoding: [0x60,0x04,0x00,0x4f]
            bdnztar-
# CHECK-BE: bctar 27, 0                  # encoding: [0x4f,0x60,0x04,0x60]
# CHECK-LE: bctar 27, 0                  # encoding: [0x60,0x04,0x60,0x4f]
            bdztar+
# CHECK-BE: bctar 26, 0                  # encoding: [0x4f,0x40,0x04,0x60]
# CHECK-LE: bctar 26, 0                  # encoding: [0x60,0x04,0x40,0x4f]
            bdztar-
