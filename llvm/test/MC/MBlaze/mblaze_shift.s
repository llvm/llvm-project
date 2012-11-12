# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to make sure that all of the TYPE-A instructions supported by
# the Microblaze can be parsed by the assembly parser.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   000000 00000 00000 00000 00000000000

# CHECK:    bsrl
# BINARY:   010001 00001 00010 00011 00000000000
# CHECK:    encoding: [0x44,0x22,0x18,0x00]
            bsrl    r1, r2, r3

# CHECK:    bsra
# BINARY:   010001 00001 00010 00011 01000000000
# CHECK:    encoding: [0x44,0x22,0x1a,0x00]
            bsra    r1, r2, r3

# CHECK:    bsll
# BINARY:   010001 00001 00010 00011 10000000000
# CHECK:    encoding: [0x44,0x22,0x1c,0x00]
            bsll    r1, r2, r3

# CHECK:    bsrli
# BINARY:   011001 00001 00010 0000000000000000
# CHECK:    encoding: [0x64,0x22,0x00,0x00]
            bsrli   r1, r2, 0

# CHECK:    bsrai
# BINARY:   011001 00001 00010 0000001000000000
# CHECK:    encoding: [0x64,0x22,0x02,0x00]
            bsrai   r1, r2, 0

# CHECK:    bslli
# BINARY:   011001 00001 00010 0000010000000000
# CHECK:    encoding: [0x64,0x22,0x04,0x00]
            bslli   r1, r2, 0

# CHECK:    sra
# BINARY:   100100 00001 00010 00000 00000000001
# CHECK:    encoding: [0x90,0x22,0x00,0x01]
            sra     r1, r2

# CHECK:    srl
# BINARY:   100100 00001 00010 00000 00001000001
# CHECK:    encoding: [0x90,0x22,0x00,0x41]
            srl     r1, r2
