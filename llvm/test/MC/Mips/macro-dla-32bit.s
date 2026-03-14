# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r6 | \
# RUN:   FileCheck %s

.set mips64r2

dla $5, 0x00000001 # CHECK: addiu $5, $zero, 1      # encoding: [0x24,0x05,0x00,0x01]
dla $5, 0x00000002 # CHECK: addiu $5, $zero, 2      # encoding: [0x24,0x05,0x00,0x02]
dla $5, 0x00004000 # CHECK: addiu $5, $zero, 16384  # encoding: [0x24,0x05,0x40,0x00]
dla $5, 0x00008000 # CHECK: ori   $5, $zero, 32768  # encoding: [0x34,0x05,0x80,0x00]
dla $5, 0xffffffff # CHECK: addiu $5, $zero, -1     # encoding: [0x24,0x05,0xff,0xff]
dla $5, 0xfffffffe # CHECK: addiu $5, $zero, -2     # encoding: [0x24,0x05,0xff,0xfe]
dla $5, 0xffffc000 # CHECK: addiu $5, $zero, -16384 # encoding: [0x24,0x05,0xc0,0x00]
dla $5, 0xffff8000 # CHECK: addiu $5, $zero, -32768 # encoding: [0x24,0x05,0x80,0x00]

dla $5, 0x00010000 # CHECK: lui $5, 1      # encoding: [0x3c,0x05,0x00,0x01]
dla $5, 0x00020000 # CHECK: lui $5, 2      # encoding: [0x3c,0x05,0x00,0x02]
dla $5, 0x40000000 # CHECK: lui $5, 16384  # encoding: [0x3c,0x05,0x40,0x00]
dla $5, 0x80000000 # CHECK: lui $5, 32768  # encoding: [0x3c,0x05,0x80,0x00]
dla $5, 0xffff0000 # CHECK: lui $5, 65535  # encoding: [0x3c,0x05,0xff,0xff]
dla $5, 0xfffe0000 # CHECK: lui $5, 65534  # encoding: [0x3c,0x05,0xff,0xfe]
dla $5, 0xc0000000 # CHECK: lui $5, 49152  # encoding: [0x3c,0x05,0xc0,0x00]
dla $5, 0x80000000 # CHECK: lui $5, 32768  # encoding: [0x3c,0x05,0x80,0x00]

dla $5, 0x00010001 # CHECK: lui $5, 1        # encoding: [0x3c,0x05,0x00,0x01]
                   # CHECK: ori $5, $5, 1    # encoding: [0x34,0xa5,0x00,0x01]
dla $5, 0x00020001 # CHECK: lui $5, 2        # encoding: [0x3c,0x05,0x00,0x02]
                   # CHECK: ori $5, $5, 1    # encoding: [0x34,0xa5,0x00,0x01]
dla $5, 0x40000001 # CHECK: lui $5, 16384    # encoding: [0x3c,0x05,0x40,0x00]
                   # CHECK: ori $5, $5, 1    # encoding: [0x34,0xa5,0x00,0x01]
dla $5, 0x80000001 # CHECK: lui $5, 32768    # encoding: [0x3c,0x05,0x80,0x00]
                   # CHECK: ori $5, $5, 1    # encoding: [0x34,0xa5,0x00,0x01]
dla $5, 0x00010002 # CHECK: lui $5, 1        # encoding: [0x3c,0x05,0x00,0x01]
                   # CHECK: ori $5, $5, 2    # encoding: [0x34,0xa5,0x00,0x02]
dla $5, 0x00020002 # CHECK: lui $5, 2        # encoding: [0x3c,0x05,0x00,0x02]
                   # CHECK: ori $5, $5, 2    # encoding: [0x34,0xa5,0x00,0x02]
dla $5, 0x40000002 # CHECK: lui $5, 16384    # encoding: [0x3c,0x05,0x40,0x00]
                   # CHECK: ori $5, $5, 2    # encoding: [0x34,0xa5,0x00,0x02]
dla $5, 0x80000002 # CHECK: lui $5, 32768    # encoding: [0x3c,0x05,0x80,0x00]
                   # CHECK: ori $5, $5, 2    # encoding: [0x34,0xa5,0x00,0x02]
dla $5, 0x00014000 # CHECK: lui $5, 1        # encoding: [0x3c,0x05,0x00,0x01]
                   # CHECK: ori $5, $5, 16384    # encoding: [0x34,0xa5,0x40,0x00]
dla $5, 0x00024000 # CHECK: lui $5, 2            # encoding: [0x3c,0x05,0x00,0x02]
                   # CHECK: ori $5, $5, 16384    # encoding: [0x34,0xa5,0x40,0x00]
dla $5, 0x40004000 # CHECK: lui $5, 16384        # encoding: [0x3c,0x05,0x40,0x00]
                   # CHECK: ori $5, $5, 16384    # encoding: [0x34,0xa5,0x40,0x00]
dla $5, 0x80004000 # CHECK: lui $5, 32768        # encoding: [0x3c,0x05,0x80,0x00]
                   # CHECK: ori $5, $5, 16384    # encoding: [0x34,0xa5,0x40,0x00]
dla $5, 0x00018000 # CHECK: lui $5, 1            # encoding: [0x3c,0x05,0x00,0x01]
                   # CHECK: ori $5, $5, 32768    # encoding: [0x34,0xa5,0x80,0x00]
dla $5, 0x00028000 # CHECK: lui $5, 2            # encoding: [0x3c,0x05,0x00,0x02]
                   # CHECK: ori $5, $5, 32768    # encoding: [0x34,0xa5,0x80,0x00]
dla $5, 0x40008000 # CHECK: lui $5, 16384        # encoding: [0x3c,0x05,0x40,0x00]
                   # CHECK: ori $5, $5, 32768    # encoding: [0x34,0xa5,0x80,0x00]
dla $5, 0x80008000 # CHECK: lui $5, 32768        # encoding: [0x3c,0x05,0x80,0x00]
                   # CHECK: ori $5, $5, 32768    # encoding: [0x34,0xa5,0x80,0x00]
dla $5, 0xffff4000 # CHECK: lui $5, 65535        # encoding: [0x3c,0x05,0xff,0xff]
                   # CHECK: ori $5, $5, 16384    # encoding: [0x34,0xa5,0x40,0x00]
dla $5, 0xfffe8000 # CHECK: lui $5, 65534        # encoding: [0x3c,0x05,0xff,0xfe]
                   # CHECK: ori $5, $5, 32768    # encoding: [0x34,0xa5,0x80,0x00]
dla $5, 0xc0008000 # CHECK: lui $5, 49152        # encoding: [0x3c,0x05,0xc0,0x00]
                   # CHECK: ori $5, $5, 32768    # encoding: [0x34,0xa5,0x80,0x00]
dla $5, 0x80008000 # CHECK: lui $5, 32768        # encoding: [0x3c,0x05,0x80,0x00]
                   # CHECK: ori $5, $5, 32768    # encoding: [0x34,0xa5,0x80,0x00]

dla $5, 0x00000001($6) # CHECK: addiu $5, $6, 1         # encoding: [0x24,0xc5,0x00,0x01]
dla $5, 0x00000002($6) # CHECK: addiu $5, $6, 2         # encoding: [0x24,0xc5,0x00,0x02]
dla $5, 0x00004000($6) # CHECK: addiu $5, $6, 16384     # encoding: [0x24,0xc5,0x40,0x00]
dla $5, 0x00008000($6) # CHECK: ori   $5, $zero, 32768  # encoding: [0x34,0x05,0x80,0x00]
                       # CHECK: addu $5, $5, $6         # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0xffffffff($6) # CHECK: addiu $5, $6, -1        # encoding: [0x24,0xc5,0xff,0xff]
dla $5, 0xfffffffe($6) # CHECK: addiu $5, $6, -2        # encoding: [0x24,0xc5,0xff,0xfe]
dla $5, 0xffffc000($6) # CHECK: addiu $5, $6, -16384    # encoding: [0x24,0xc5,0xc0,0x00]
dla $5, 0xffff8000($6) # CHECK: addiu $5, $6, -32768    # encoding: [0x24,0xc5,0x80,0x00]

dla $5, 0x00010000($6) # CHECK: lui $5, 1       # encoding: [0x3c,0x05,0x00,0x01]
                       # CHECK: addu $5, $5, $6 # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x00020000($6) # CHECK: lui $5, 2       # encoding: [0x3c,0x05,0x00,0x02]
                       # CHECK: addu $5, $5, $6 # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x40000000($6) # CHECK: lui $5, 16384   # encoding: [0x3c,0x05,0x40,0x00]
                       # CHECK: addu $5, $5, $6 # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x80000000($6) # CHECK: lui $5, 32768   # encoding: [0x3c,0x05,0x80,0x00]
                       # CHECK: addu $5, $5, $6 # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0xffff0000($6) # CHECK: lui $5, 65535   # encoding: [0x3c,0x05,0xff,0xff]
                       # CHECK: addu $5, $5, $6 # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0xfffe0000($6) # CHECK: lui $5, 65534   # encoding: [0x3c,0x05,0xff,0xfe]
                       # CHECK: addu $5, $5, $6 # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0xc0000000($6) # CHECK: lui $5, 49152   # encoding: [0x3c,0x05,0xc0,0x00]
                       # CHECK: addu $5, $5, $6 # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x80000000($6) # CHECK: lui $5, 32768   # encoding: [0x3c,0x05,0x80,0x00]
                       # CHECK: addu $5, $5, $6 # encoding: [0x00,0xa6,0x28,0x21]

dla $5, 0x00010001($6) # CHECK: lui $5, 1         # encoding: [0x3c,0x05,0x00,0x01]
                       # CHECK: ori $5, $5, 1     # encoding: [0x34,0xa5,0x00,0x01]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x00020001($6) # CHECK: lui $5, 2         # encoding: [0x3c,0x05,0x00,0x02]
                       # CHECK: ori $5, $5, 1     # encoding: [0x34,0xa5,0x00,0x01]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x40000001($6) # CHECK: lui $5, 16384     # encoding: [0x3c,0x05,0x40,0x00]
                       # CHECK: ori $5, $5, 1     # encoding: [0x34,0xa5,0x00,0x01]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x80000001($6) # CHECK: lui $5, 32768     # encoding: [0x3c,0x05,0x80,0x00]
                       # CHECK: ori $5, $5, 1     # encoding: [0x34,0xa5,0x00,0x01]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x00010002($6) # CHECK: lui $5, 1         # encoding: [0x3c,0x05,0x00,0x01]
                       # CHECK: ori $5, $5, 2     # encoding: [0x34,0xa5,0x00,0x02]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x00020002($6) # CHECK: lui $5, 2         # encoding: [0x3c,0x05,0x00,0x02]
                       # CHECK: ori $5, $5, 2     # encoding: [0x34,0xa5,0x00,0x02]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x40000002($6) # CHECK: lui $5, 16384     # encoding: [0x3c,0x05,0x40,0x00]
                       # CHECK: ori $5, $5, 2     # encoding: [0x34,0xa5,0x00,0x02]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x80000002($6) # CHECK: lui $5, 32768     # encoding: [0x3c,0x05,0x80,0x00]
                       # CHECK: ori $5, $5, 2     # encoding: [0x34,0xa5,0x00,0x02]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x00014000($6) # CHECK: lui $5, 1         # encoding: [0x3c,0x05,0x00,0x01]
                       # CHECK: ori $5, $5, 16384 # encoding: [0x34,0xa5,0x40,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x00024000($6) # CHECK: lui $5, 2         # encoding: [0x3c,0x05,0x00,0x02]
                       # CHECK: ori $5, $5, 16384 # encoding: [0x34,0xa5,0x40,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x40004000($6) # CHECK: lui $5, 16384     # encoding: [0x3c,0x05,0x40,0x00]
                       # CHECK: ori $5, $5, 16384 # encoding: [0x34,0xa5,0x40,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x80004000($6) # CHECK: lui $5, 32768     # encoding: [0x3c,0x05,0x80,0x00]
                       # CHECK: ori $5, $5, 16384 # encoding: [0x34,0xa5,0x40,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x00018000($6) # CHECK: lui $5, 1         # encoding: [0x3c,0x05,0x00,0x01]
                       # CHECK: ori $5, $5, 32768 # encoding: [0x34,0xa5,0x80,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x00028000($6) # CHECK: lui $5, 2         # encoding: [0x3c,0x05,0x00,0x02]
                       # CHECK: ori $5, $5, 32768 # encoding: [0x34,0xa5,0x80,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x40008000($6) # CHECK: lui $5, 16384     # encoding: [0x3c,0x05,0x40,0x00]
                       # CHECK: ori $5, $5, 32768 # encoding: [0x34,0xa5,0x80,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x80008000($6) # CHECK: lui $5, 32768     # encoding: [0x3c,0x05,0x80,0x00]
                       # CHECK: ori $5, $5, 32768 # encoding: [0x34,0xa5,0x80,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0xffff4000($6) # CHECK: lui $5, 65535     # encoding: [0x3c,0x05,0xff,0xff]
                       # CHECK: ori $5, $5, 16384 # encoding: [0x34,0xa5,0x40,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0xfffe8000($6) # CHECK: lui $5, 65534     # encoding: [0x3c,0x05,0xff,0xfe]
                       # CHECK: ori $5, $5, 32768 # encoding: [0x34,0xa5,0x80,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0xc0008000($6) # CHECK: lui $5, 49152     # encoding: [0x3c,0x05,0xc0,0x00]
                       # CHECK: ori $5, $5, 32768 # encoding: [0x34,0xa5,0x80,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
dla $5, 0x80008000($6) # CHECK: lui $5, 32768     # encoding: [0x3c,0x05,0x80,0x00]
                       # CHECK: ori $5, $5, 32768 # encoding: [0x34,0xa5,0x80,0x00]
                       # CHECK: addu $5, $5, $6   # encoding: [0x00,0xa6,0x28,0x21]
# There are no further interesting immediates.

symbol:           # CHECK-LABEL: symbol:
.extern extern_sym
.option pic0      # CHECK-LABEL: .option pic0
dla $5, extern_sym

# CHECK: lui $5, %hi(extern_sym)         # encoding: [0x3c,0x05,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %hi(extern_sym), kind: fixup_Mips_HI16
# CHECK: addiu $5, $5, %lo(extern_sym)   # encoding: [0x24,0xa5,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %lo(extern_sym), kind: fixup_Mips_LO16

dla $5, extern_sym($8)

# CHECK: lui $5, %hi(extern_sym)         # encoding: [0x3c,0x05,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %hi(extern_sym), kind: fixup_Mips_HI16
# CHECK: addiu $5, $5, %lo(extern_sym)   # encoding: [0x24,0xa5,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %lo(extern_sym), kind: fixup_Mips_LO16
# CHECK: addu $5, $5, $8                 # encoding: [0x00,0xa8,0x28,0x21]

dla $5, extern_sym($5)

# CHECK: lui $1, %hi(extern_sym)         # encoding: [0x3c,0x01,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %hi(extern_sym), kind: fixup_Mips_HI16
# CHECK: addiu $1, $1, %lo(extern_sym)   # encoding: [0x24,0x21,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %lo(extern_sym), kind: fixup_Mips_LO16
# CHECK: addu $5, $1, $5                 # encoding: [0x00,0x25,0x28,0x21]

dla $5, extern_sym+8

# CHECK: lui $5, %hi(extern_sym+8)       # encoding: [0x3c,0x05,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %hi(extern_sym+8), kind: fixup_Mips_HI16
# CHECK: addiu $5, $5, %lo(extern_sym+8) # encoding: [0x24,0xa5,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %lo(extern_sym+8), kind: fixup_Mips_LO16

dla $5, extern_sym+8($8)

# CHECK: lui $5, %hi(extern_sym+8)       # encoding: [0x3c,0x05,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %hi(extern_sym+8), kind: fixup_Mips_HI16
# CHECK: addiu $5, $5, %lo(extern_sym+8) # encoding: [0x24,0xa5,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %lo(extern_sym+8), kind: fixup_Mips_LO16
# CHECK: addu $5, $5, $8                 # encoding: [0x00,0xa8,0x28,0x21]

dla $5, extern_sym-8($5)

# CHECK: lui $1, %hi(extern_sym-8)       # encoding: [0x3c,0x01,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %hi(extern_sym-8), kind: fixup_Mips_HI16
# CHECK: addiu $1, $1, %lo(extern_sym-8) # encoding: [0x24,0x21,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %lo(extern_sym-8), kind: fixup_Mips_LO16
# CHECK: addu $5, $1, $5                 # encoding: [0x00,0x25,0x28,0x21]

dla $5, extern_sym-8

# CHECK: lui $5, %hi(extern_sym-8)       # encoding: [0x3c,0x05,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %hi(extern_sym-8), kind: fixup_Mips_HI16
# CHECK: addiu $5, $5, %lo(extern_sym-8) # encoding: [0x24,0xa5,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %lo(extern_sym-8), kind: fixup_Mips_LO16

dla $5, extern_sym-8($8)

# CHECK: lui $5, %hi(extern_sym-8)       # encoding: [0x3c,0x05,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %hi(extern_sym-8), kind: fixup_Mips_HI16
# CHECK: addiu $5, $5, %lo(extern_sym-8) # encoding: [0x24,0xa5,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %lo(extern_sym-8), kind: fixup_Mips_LO16
# CHECK: addu $5, $5, $8                 # encoding: [0x00,0xa8,0x28,0x21]

dla $5, extern_sym-8($5)

# CHECK: lui $1, %hi(extern_sym-8)       # encoding: [0x3c,0x01,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %hi(extern_sym-8), kind: fixup_Mips_HI16
# CHECK: addiu $1, $1, %lo(extern_sym-8) # encoding: [0x24,0x21,A,A]
# CHECK:                                 #   fixup A - offset: 0, value: %lo(extern_sym-8), kind: fixup_Mips_LO16
# CHECK: addu $5, $1, $5                 # encoding: [0x00,0x25,0x28,0x21]

.option pic2
