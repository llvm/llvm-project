# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to make sure that all of the TYPE-A instructions supported by
# the Microblaze can be parsed by the assembly parser.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   000000 00000 00000 00000 00000000000

# CHECK:    lbu
# BINARY:   110000 00001 00010 00011 00000000000
# CHECK:    encoding: [0xc0,0x22,0x18,0x00]
            lbu     r1, r2, r3

# CHECK:    lbur
# BINARY:   110000 00001 00010 00011 01000000000
# CHECK:    encoding: [0xc0,0x22,0x1a,0x00]
            lbur    r1, r2, r3

# CHECK:    lbui
# BINARY:   111000 00001 00010 0000000000011100
# CHECK:    encoding: [0xe0,0x22,0x00,0x1c]
            lbui    r1, r2, 28

# CHECK:    lhu
# BINARY:   110001 00001 00010 00011 00000000000
# CHECK:    encoding: [0xc4,0x22,0x18,0x00]
            lhu     r1, r2, r3

# CHECK:    lhur
# BINARY:   110001 00001 00010 00011 01000000000
# CHECK:    encoding: [0xc4,0x22,0x1a,0x00]
            lhur    r1, r2, r3

# CHECK:    lhui
# BINARY:   111001 00001 00010 0000000000011100
# CHECK:    encoding: [0xe4,0x22,0x00,0x1c]
            lhui    r1, r2, 28

# CHECK:    lw
# BINARY:   110010 00001 00010 00011 00000000000
# CHECK:    encoding: [0xc8,0x22,0x18,0x00]
            lw      r1, r2, r3

# CHECK:    lwr
# BINARY:   110010 00001 00010 00011 01000000000
# CHECK:    encoding: [0xc8,0x22,0x1a,0x00]
            lwr    r1, r2, r3

# CHECK:    lwi
# BINARY:   111010 00001 00010 0000000000011100
# CHECK:    encoding: [0xe8,0x22,0x00,0x1c]
            lwi     r1, r2, 28

# CHECK:    lwx
# BINARY:   110010 00001 00010 00011 10000000000
# CHECK:    encoding: [0xc8,0x22,0x1c,0x00]
            lwx      r1, r2, r3

# CHECK:    sb
# BINARY:   110100 00001 00010 00011 00000000000
# CHECK:    encoding: [0xd0,0x22,0x18,0x00]
            sb      r1, r2, r3

# CHECK:    sbr
# BINARY:   110100 00001 00010 00011 01000000000
# CHECK:    encoding: [0xd0,0x22,0x1a,0x00]
            sbr     r1, r2, r3

# CHECK:    sbi
# BINARY:   111100 00001 00010 0000000000011100
# CHECK:    encoding: [0xf0,0x22,0x00,0x1c]
            sbi     r1, r2, 28

# CHECK:    sh
# BINARY:   110101 00001 00010 00011 00000000000
# CHECK:    encoding: [0xd4,0x22,0x18,0x00]
            sh      r1, r2, r3

# CHECK:    shr
# BINARY:   110101 00001 00010 00011 01000000000
# CHECK:    encoding: [0xd4,0x22,0x1a,0x00]
            shr     r1, r2, r3

# CHECK:    shi
# BINARY:   111101 00001 00010 0000000000011100
# CHECK:    encoding: [0xf4,0x22,0x00,0x1c]
            shi     r1, r2, 28

# CHECK:    sw
# BINARY:   110110 00001 00010 00011 00000000000
# CHECK:    encoding: [0xd8,0x22,0x18,0x00]
            sw      r1, r2, r3

# CHECK:    swr
# BINARY:   110110 00001 00010 00011 01000000000
# CHECK:    encoding: [0xd8,0x22,0x1a,0x00]
            swr    r1, r2, r3

# CHECK:    swi
# BINARY:   111110 00001 00010 0000000000011100
# CHECK:    encoding: [0xf8,0x22,0x00,0x1c]
            swi     r1, r2, 28

# CHECK:    swx
# BINARY:   110110 00001 00010 00011 10000000000
# CHECK:    encoding: [0xd8,0x22,0x1c,0x00]
            swx      r1, r2, r3
