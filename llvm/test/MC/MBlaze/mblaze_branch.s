# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to make sure that all of the TYPE-A instructions supported by
# the Microblaze can be parsed by the assembly parser.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   000000 00000 00000 00000 00000000000

# CHECK:    beq
# BINARY:   100111 00000 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x02,0x18,0x00]
            beq     r2, r3

# CHECK:    bge
# BINARY:   100111 00101 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0xa2,0x18,0x00]
            bge     r2, r3

# CHECK:    bgt
# BINARY:   100111 00100 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x82,0x18,0x00]
            bgt     r2, r3

# CHECK:    ble
# BINARY:   100111 00011 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x62,0x18,0x00]
            ble     r2, r3

# CHECK:    blt
# BINARY:   100111 00010 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x42,0x18,0x00]
            blt     r2, r3

# CHECK:    bne
# BINARY:   100111 00001 00010 00011 00000000000
# CHECK:    encoding: [0x9c,0x22,0x18,0x00]
            bne     r2, r3

# CHECK:    beqd
# BINARY:   100111 10000 00010 00011 00000000000
# CHECK:    encoding: [0x9e,0x02,0x18,0x00]
            beqd    r2, r3

# CHECK:    bged
# BINARY:   100111 10101 00010 00011 00000000000
# CHECK:    encoding: [0x9e,0xa2,0x18,0x00]
            bged    r2, r3

# CHECK:    bgtd
# BINARY:   100111 10100 00010 00011 00000000000
# CHECK:    encoding: [0x9e,0x82,0x18,0x00]
            bgtd    r2, r3

# CHECK:    bled
# BINARY:   100111 10011 00010 00011 00000000000
# CHECK:    encoding: [0x9e,0x62,0x18,0x00]
            bled    r2, r3

# CHECK:    bltd
# BINARY:   100111 10010 00010 00011 00000000000
# CHECK:    encoding: [0x9e,0x42,0x18,0x00]
            bltd    r2, r3

# CHECK:    bned
# BINARY:   100111 10001 00010 00011 00000000000
# CHECK:    encoding: [0x9e,0x22,0x18,0x00]
            bned    r2, r3

# CHECK:    br
# BINARY:   100110 00000 00000 00011 00000000000
# CHECK:    encoding: [0x98,0x00,0x18,0x00]
            br      r3

# CHECK:    bra
# BINARY:   100110 00000 01000 00011 00000000000
# CHECK:    encoding: [0x98,0x08,0x18,0x00]
            bra     r3

# CHECK:    brd
# BINARY:   100110 00000 10000 00011 00000000000
# CHECK:    encoding: [0x98,0x10,0x18,0x00]
            brd     r3

# CHECK:    brad
# BINARY:   100110 00000 11000 00011 00000000000
# CHECK:    encoding: [0x98,0x18,0x18,0x00]
            brad    r3

# CHECK:    brld
# BINARY:   100110 01111 10100 00011 00000000000
# CHECK:    encoding: [0x99,0xf4,0x18,0x00]
            brld    r15, r3

# CHECK:    brald
# BINARY:   100110 01111 11100 00011 00000000000
# CHECK:    encoding: [0x99,0xfc,0x18,0x00]
            brald   r15, r3

# CHECK:    brk
# BINARY:   100110 01111 01100 00011 00000000000
# CHECK:    encoding: [0x99,0xec,0x18,0x00]
            brk     r15, r3

# CHECK:    beqi
# BINARY:   101111 00000 00010 0000000000000000
# CHECK:    encoding: [0xbc,0x02,0x00,0x00]
            beqi    r2, 0

# CHECK:    bgei
# BINARY:   101111 00101 00010 0000000000000000
# CHECK:    encoding: [0xbc,0xa2,0x00,0x00]
            bgei    r2, 0

# CHECK:    bgti
# BINARY:   101111 00100 00010 0000000000000000
# CHECK:    encoding: [0xbc,0x82,0x00,0x00]
            bgti    r2, 0

# CHECK:    blei
# BINARY:   101111 00011 00010 0000000000000000
# CHECK:    encoding: [0xbc,0x62,0x00,0x00]
            blei    r2, 0

# CHECK:    blti
# BINARY:   101111 00010 00010 0000000000000000
# CHECK:    encoding: [0xbc,0x42,0x00,0x00]
            blti    r2, 0

# CHECK:    bnei
# BINARY:   101111 00001 00010 0000000000000000
# CHECK:    encoding: [0xbc,0x22,0x00,0x00]
            bnei    r2, 0

# CHECK:    beqid
# BINARY:   101111 10000 00010 0000000000000000
# CHECK:    encoding: [0xbe,0x02,0x00,0x00]
            beqid   r2, 0

# CHECK:    bgeid
# BINARY:   101111 10101 00010 0000000000000000
# CHECK:    encoding: [0xbe,0xa2,0x00,0x00]
            bgeid   r2, 0

# CHECK:    bgtid
# BINARY:   101111 10100 00010 0000000000000000
# CHECK:    encoding: [0xbe,0x82,0x00,0x00]
            bgtid   r2, 0

# CHECK:    bleid
# BINARY:   101111 10011 00010 0000000000000000
# CHECK:    encoding: [0xbe,0x62,0x00,0x00]
            bleid   r2, 0

# CHECK:    bltid
# BINARY:   101111 10010 00010 0000000000000000
# CHECK:    encoding: [0xbe,0x42,0x00,0x00]
            bltid   r2, 0

# CHECK:    bneid
# BINARY:   101111 10001 00010 0000000000000000
# CHECK:    encoding: [0xbe,0x22,0x00,0x00]
            bneid   r2, 0

# CHECK:    bri
# BINARY:   101110 00000 00000 0000000000000000
# CHECK:    encoding: [0xb8,0x00,0x00,0x00]
            bri     0

# CHECK:    brai
# BINARY:   101110 00000 01000 0000000000000000
# CHECK:    encoding: [0xb8,0x08,0x00,0x00]
            brai    0

# CHECK:    brid
# BINARY:   101110 00000 10000 0000000000000000
# CHECK:    encoding: [0xb8,0x10,0x00,0x00]
            brid    0

# CHECK:    braid
# BINARY:   101110 00000 11000 0000000000000000
# CHECK:    encoding: [0xb8,0x18,0x00,0x00]
            braid   0

# CHECK:    brlid
# BINARY:   101110 01111 10100 0000000000000000
# CHECK:    encoding: [0xb9,0xf4,0x00,0x00]
            brlid   r15, 0

# CHECK:    bralid
# BINARY:   101110 01111 11100 0000000000000000
# CHECK:    encoding: [0xb9,0xfc,0x00,0x00]
            bralid  r15, 0

# CHECK:    brki
# BINARY:   101110 01111 01100 0000000000000000
# CHECK:    encoding: [0xb9,0xec,0x00,0x00]
            brki    r15, 0
