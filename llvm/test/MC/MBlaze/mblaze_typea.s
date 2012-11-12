# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to make sure that all of the TYPE-A instructions supported by
# the Microblaze can be parsed by the assembly parser.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   000000 00000 00000 00000 00000000000

# CHECK:    add
# BINARY:   000000 00001 00010 00011 00000000000
# CHECK:    encoding: [0x00,0x22,0x18,0x00]
            add     r1, r2, r3

# CHECK:    addc
# BINARY:   000010 00001 00010 00011 00000000000
# CHECK:    encoding: [0x08,0x22,0x18,0x00]
            addc    r1, r2, r3

# CHECK:    addk
# BINARY:   000100 00001 00010 00011 00000000000
# CHECK:    encoding: [0x10,0x22,0x18,0x00]
            addk    r1, r2, r3

# CHECK:    addkc
# BINARY:   000110 00001 00010 00011 00000000000
# CHECK:    encoding: [0x18,0x22,0x18,0x00]
            addkc   r1, r2, r3

# CHECK:    and
# BINARY:   100001 00001 00010 00011 00000000000
# CHECK:    encoding: [0x84,0x22,0x18,0x00]
            and     r1, r2, r3

# CHECK:    andn
# BINARY:   100011 00001 00010 00011 00000000000
# CHECK:    encoding: [0x8c,0x22,0x18,0x00]
            andn    r1, r2, r3

# CHECK:    cmp
# BINARY:   000101 00001 00010 00011 00000000001
# CHECK:    encoding: [0x14,0x22,0x18,0x01]
            cmp     r1, r2, r3

# CHECK:    cmpu
# BINARY:   000101 00001 00010 00011 00000000011
# CHECK:    encoding: [0x14,0x22,0x18,0x03]
            cmpu    r1, r2, r3

# CHECK:    idiv
# BINARY:   010010 00001 00010 00011 00000000000
# CHECK:    encoding: [0x48,0x22,0x18,0x00]
            idiv    r1, r2, r3

# CHECK:    idivu
# BINARY:   010010 00001 00010 00011 00000000010
# CHECK:    encoding: [0x48,0x22,0x18,0x02]
            idivu   r1, r2, r3

# CHECK:    mul
# BINARY:   010000 00001 00010 00011 00000000000
# CHECK:    encoding: [0x40,0x22,0x18,0x00]
            mul    r1, r2, r3

# CHECK:    mulh
# BINARY:   010000 00001 00010 00011 00000000001
# CHECK:    encoding: [0x40,0x22,0x18,0x01]
            mulh   r1, r2, r3

# CHECK:    mulhu
# BINARY:   010000 00001 00010 00011 00000000011
# CHECK:    encoding: [0x40,0x22,0x18,0x03]
            mulhu  r1, r2, r3

# CHECK:    mulhsu
# BINARY:   010000 00001 00010 00011 00000000010
# CHECK:    encoding: [0x40,0x22,0x18,0x02]
            mulhsu r1, r2, r3

# CHECK:    or
# BINARY:   100000 00001 00010 00011 00000000000
# CHECK:    encoding: [0x80,0x22,0x18,0x00]
            or      r1, r2, r3

# CHECK:    rsub
# BINARY:   000001 00001 00010 00011 00000000000
# CHECK:    encoding: [0x04,0x22,0x18,0x00]
            rsub    r1, r2, r3

# CHECK:    rsubc
# BINARY:   000011 00001 00010 00011 00000000000
# CHECK:    encoding: [0x0c,0x22,0x18,0x00]
            rsubc   r1, r2, r3

# CHECK:    rsubk
# BINARY:   000101 00001 00010 00011 00000000000
# CHECK:    encoding: [0x14,0x22,0x18,0x00]
            rsubk   r1, r2, r3

# CHECK:    rsubkc
# BINARY:   000111 00001 00010 00011 00000000000
# CHECK:    encoding: [0x1c,0x22,0x18,0x00]
            rsubkc  r1, r2, r3

# CHECK:    sext16
# BINARY:   100100 00001 00010 00000 00001100001
# CHECK:    encoding: [0x90,0x22,0x00,0x61]
            sext16  r1, r2

# CHECK:    sext8
# BINARY:   100100 00001 00010 00000 00001100000
# CHECK:    encoding: [0x90,0x22,0x00,0x60]
            sext8   r1, r2

# CHECK:    xor
# BINARY:   100010 00001 00010 00011 00000000000
# CHECK:    encoding: [0x88,0x22,0x18,0x00]
            xor     r1, r2, r3

# CHECK:    nop
# BINARY:   100000 00000 00000 00000 00000000000
# CHECK:    encoding: [0x80,0x00,0x00,0x00]
        nop
