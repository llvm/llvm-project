# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# In the microblaze instruction set, any TYPE-B instruction with a
# signed immediate value requiring more than 16-bits must be prefixed
# with an IMM instruction that contains the high 16-bits. The higher
# 16-bits are then combined with the lower 16-bits in the original
# instruction to form a 32-bit immediate value.
#
# The generation of IMM instructions is handled automatically by the
# code emitter. Test to ensure that IMM instructions are generated
# when they are suppose to and are not generated when they are not
# needed.

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000000000
# CHECK:    encoding: [0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x00000000

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000000001
# CHECK:    encoding: [0x20,0x00,0x00,0x01]
            addi    r0, r0, 0x00000001

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000000010
# CHECK:    encoding: [0x20,0x00,0x00,0x02]
            addi    r0, r0, 0x00000002

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000000100
# CHECK:    encoding: [0x20,0x00,0x00,0x04]
            addi    r0, r0, 0x00000004

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000001000
# CHECK:    encoding: [0x20,0x00,0x00,0x08]
            addi    r0, r0, 0x00000008

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000010000
# CHECK:    encoding: [0x20,0x00,0x00,0x10]
            addi    r0, r0, 0x00000010

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000000100000
# CHECK:    encoding: [0x20,0x00,0x00,0x20]
            addi    r0, r0, 0x00000020

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000001000000
# CHECK:    encoding: [0x20,0x00,0x00,0x40]
            addi    r0, r0, 0x00000040

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000010000000
# CHECK:    encoding: [0x20,0x00,0x00,0x80]
            addi    r0, r0, 0x00000080

# CHECK:    addi
# BINARY:   001000 00000 00000 0000000100000000
# CHECK:    encoding: [0x20,0x00,0x01,0x00]
            addi    r0, r0, 0x00000100

# CHECK:    addi
# BINARY:   001000 00000 00000 0000001000000000
# CHECK:    encoding: [0x20,0x00,0x02,0x00]
            addi    r0, r0, 0x00000200

# CHECK:    addi
# BINARY:   001000 00000 00000 0000010000000000
# CHECK:    encoding: [0x20,0x00,0x04,0x00]
            addi    r0, r0, 0x00000400

# CHECK:    addi
# BINARY:   001000 00000 00000 0000100000000000
# CHECK:    encoding: [0x20,0x00,0x08,0x00]
            addi    r0, r0, 0x00000800

# CHECK:    addi
# BINARY:   001000 00000 00000 0001000000000000
# CHECK:    encoding: [0x20,0x00,0x10,0x00]
            addi    r0, r0, 0x00001000

# CHECK:    addi
# BINARY:   001000 00000 00000 0010000000000000
# CHECK:    encoding: [0x20,0x00,0x20,0x00]
            addi    r0, r0, 0x00002000

# CHECK:    addi
# BINARY:   001000 00000 00000 0100000000000000
# CHECK:    encoding: [0x20,0x00,0x40,0x00]
            addi    r0, r0, 0x00004000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000000000000
# BINARY:   001000 00000 00000 1000000000000000
# CHECK:    encoding: [0xb0,0x00,0x00,0x00,0x20,0x00,0x80,0x00]
            addi    r0, r0, 0x00008000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000000000001
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x00,0x01,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x00010000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000000000010
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x00,0x02,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x00020000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000000000100
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x00,0x04,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x00040000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000000001000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x00,0x08,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x00080000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000000010000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x00,0x10,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x00100000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000000100000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x00,0x20,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x00200000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000001000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x00,0x40,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x00400000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000010000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x00,0x80,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x00800000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000000100000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x01,0x00,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x01000000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000001000000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x02,0x00,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x02000000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000010000000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x04,0x00,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x04000000

# CHECK:    addi
# BINARY:   101100 00000 00000 0000100000000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x08,0x00,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x08000000

# CHECK:    addi
# BINARY:   101100 00000 00000 0001000000000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x10,0x00,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x10000000

# CHECK:    addi
# BINARY:   101100 00000 00000 0010000000000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x20,0x00,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x20000000

# CHECK:    addi
# BINARY:   101100 00000 00000 0100000000000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x40,0x00,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x40000000

# CHECK:    addi
# BINARY:   101100 00000 00000 1000000000000000
#           001000 00000 00000 0000000000000000
# CHECK:    encoding: [0xb0,0x00,0x80,0x00,0x20,0x00,0x00,0x00]
            addi    r0, r0, 0x80000000
