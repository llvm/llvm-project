
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Register operands

# CHECK-BE: add 1, 2, 3                     # encoding: [0x7c,0x22,0x1a,0x14]
# CHECK-LE: add 1, 2, 3                     # encoding: [0x14,0x1a,0x22,0x7c]
            add 1, 2, 3

# CHECK-BE: add 1, 2, 3                     # encoding: [0x7c,0x22,0x1a,0x14]
# CHECK-LE: add 1, 2, 3                     # encoding: [0x14,0x1a,0x22,0x7c]
            add %r1, %r2, %r3

# CHECK-BE: add 0, 0, 0                     # encoding: [0x7c,0x00,0x02,0x14]
# CHECK-LE: add 0, 0, 0                     # encoding: [0x14,0x02,0x00,0x7c]
            add 0, 0, 0

# CHECK-BE: add 31, 31, 31                  # encoding: [0x7f,0xff,0xfa,0x14]
# CHECK-LE: add 31, 31, 31                  # encoding: [0x14,0xfa,0xff,0x7f]
            add 31, 31, 31

# CHECK-BE: li 1, 0                         # encoding: [0x38,0x20,0x00,0x00]
# CHECK-LE: li 1, 0                         # encoding: [0x00,0x00,0x20,0x38]
            addi 1, 0, 0

# CHECK-BE: li 1, 0                         # encoding: [0x38,0x20,0x00,0x00]
# CHECK-LE: li 1, 0                         # encoding: [0x00,0x00,0x20,0x38]
            addi 1, %r0, 0

# Signed 16-bit immediate operands

# CHECK-BE: addi 1, 2, 0                    # encoding: [0x38,0x22,0x00,0x00]
# CHECK-LE: addi 1, 2, 0                    # encoding: [0x00,0x00,0x22,0x38]
            addi 1, 2, 0

# CHECK-BE: li 1, -32768                    # encoding: [0x38,0x20,0x80,0x00]
# CHECK-LE: li 1, -32768                    # encoding: [0x00,0x80,0x20,0x38]
            addi 1, 0, -32768

# CHECK-BE: li 1, 32767                     # encoding: [0x38,0x20,0x7f,0xff]
# CHECK-LE: li 1, 32767                     # encoding: [0xff,0x7f,0x20,0x38]
            addi 1, 0, 32767

# Unsigned 16-bit immediate operands

# CHECK-BE: ori 1, 2, 0                     # encoding: [0x60,0x41,0x00,0x00]
# CHECK-LE: ori 1, 2, 0                     # encoding: [0x00,0x00,0x41,0x60]
            ori 1, 2, 0

# CHECK-BE: ori 1, 2, 65535                 # encoding: [0x60,0x41,0xff,0xff]
# CHECK-LE: ori 1, 2, 65535                 # encoding: [0xff,0xff,0x41,0x60]
            ori 1, 2, 65535

# Signed 16-bit immediate operands (extended range for addis)

# CHECK-BE: lis 1, 0                        # encoding: [0x3c,0x20,0x00,0x00]
# CHECK-LE: lis 1, 0                        # encoding: [0x00,0x00,0x20,0x3c]
            addis 1, 0, -65536

# CHECK-BE: lis 1, -1                       # encoding: [0x3c,0x20,0xff,0xff]
# CHECK-LE: lis 1, -1                       # encoding: [0xff,0xff,0x20,0x3c]
            addis 1, 0, 65535

# D-Form memory operands

# CHECK-BE: lwz 1, 0(0)                     # encoding: [0x80,0x20,0x00,0x00]
# CHECK-LE: lwz 1, 0(0)                     # encoding: [0x00,0x00,0x20,0x80]
            lwz 1, 0(0)

# CHECK-BE: lwz 1, 0(0)                     # encoding: [0x80,0x20,0x00,0x00]
# CHECK-LE: lwz 1, 0(0)                     # encoding: [0x00,0x00,0x20,0x80]
            lwz 1, 0(%r0)

# CHECK-BE: lwz 1, 0(31)                    # encoding: [0x80,0x3f,0x00,0x00]
# CHECK-LE: lwz 1, 0(31)                    # encoding: [0x00,0x00,0x3f,0x80]
            lwz 1, 0(31)

# CHECK-BE: lwz 1, 0(31)                    # encoding: [0x80,0x3f,0x00,0x00]
# CHECK-LE: lwz 1, 0(31)                    # encoding: [0x00,0x00,0x3f,0x80]
            lwz 1, 0(%r31)

# CHECK-BE: lwz 1, -32768(2)                # encoding: [0x80,0x22,0x80,0x00]
# CHECK-LE: lwz 1, -32768(2)                # encoding: [0x00,0x80,0x22,0x80]
            lwz 1, -32768(2)

# CHECK-BE: lwz 1, 32767(2)                 # encoding: [0x80,0x22,0x7f,0xff]
# CHECK-LE: lwz 1, 32767(2)                 # encoding: [0xff,0x7f,0x22,0x80]
            lwz 1, 32767(2)


# CHECK-BE: ld 1, 0(0)                      # encoding: [0xe8,0x20,0x00,0x00]
# CHECK-LE: ld 1, 0(0)                      # encoding: [0x00,0x00,0x20,0xe8]
            ld 1, 0(0)

# CHECK-BE: ld 1, 0(0)                      # encoding: [0xe8,0x20,0x00,0x00]
# CHECK-LE: ld 1, 0(0)                      # encoding: [0x00,0x00,0x20,0xe8]
            ld 1, 0(%r0)

# CHECK-BE: ld 1, 0(31)                     # encoding: [0xe8,0x3f,0x00,0x00]
# CHECK-LE: ld 1, 0(31)                     # encoding: [0x00,0x00,0x3f,0xe8]
            ld 1, 0(31)

# CHECK-BE: ld 1, 0(31)                     # encoding: [0xe8,0x3f,0x00,0x00]
# CHECK-LE: ld 1, 0(31)                     # encoding: [0x00,0x00,0x3f,0xe8]
            ld 1, 0(%r31)

# CHECK-BE: ld 1, -32768(2)                 # encoding: [0xe8,0x22,0x80,0x00]
# CHECK-LE: ld 1, -32768(2)                 # encoding: [0x00,0x80,0x22,0xe8]
            ld 1, -32768(2)

# CHECK-BE: ld 1, 32764(2)                  # encoding: [0xe8,0x22,0x7f,0xfc]
# CHECK-LE: ld 1, 32764(2)                  # encoding: [0xfc,0x7f,0x22,0xe8]
            ld 1, 32764(2)

# CHECK-BE: ld 1, 4(2)                      # encoding: [0xe8,0x22,0x00,0x04]
# CHECK-LE: ld 1, 4(2)                      # encoding: [0x04,0x00,0x22,0xe8]
            ld 1, 4(2)

# CHECK-BE: ld 1, -4(2)                     # encoding: [0xe8,0x22,0xff,0xfc]
# CHECK-LE: ld 1, -4(2)                     # encoding: [0xfc,0xff,0x22,0xe8]
            ld 1, -4(2)


# Immediate branch operands

# CHECK-BE: b .+1024                        # encoding: [0x48,0x00,0x04,0x00]
# CHECK-LE: b .+1024                        # encoding: [0x00,0x04,0x00,0x48]
            b 1024

# CHECK-BE: ba 0x400                        # encoding: [0x48,0x00,0x04,0x02]
# CHECK-LE: ba 0x400                        # encoding: [0x02,0x04,0x00,0x48]
            ba 1024

# CHECK-BE: beq 0, .+1024                   # encoding: [0x41,0x82,0x04,0x00]
# CHECK-LE: beq 0, .+1024                   # encoding: [0x00,0x04,0x82,0x41]
            beq 1024

# CHECK-BE: beqa 0, 0x400                   # encoding: [0x41,0x82,0x04,0x02]
# CHECK-LE: beqa 0, 0x400                   # encoding: [0x02,0x04,0x82,0x41]
            beqa 1024

# CHECK-BE:                                 # encoding: [0x42,0x9f,A,0bAAAAAA01]
# CHECK-LE:                                 # encoding: [0bAAAAAA01,A,0x9f,0x42]
            bcl 20, 31, $+4

# CHECK-BE:                                 # encoding: [0x42,0x00,A,0bAAAAAA00]
# CHECK-LE:                                 # encoding: [0bAAAAAA00,A,0x00,0x42]
            bdnz $-8

# CHECK-BE: andi. 0, 3, 32767               # encoding: [0x70,0x60,0x7f,0xff]
# CHECK-LE: andi. 0, 3, 32767               # encoding: [0xff,0x7f,0x60,0x70]
            andi. %r0,%r3,~0x8000@l

# CHECK-BE: andi. 0, 3, 0                   # encoding: [0x70,0x60,0x00,0x00]
# CHECK-LE: andi. 0, 3, 0                   # encoding: [0x00,0x00,0x60,0x70]
            andi. %r0,%r3,!0x8000@l

