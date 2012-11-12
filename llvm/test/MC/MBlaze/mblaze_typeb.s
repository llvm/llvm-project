# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to make sure that all of the TYPE-B instructions supported by
# the Microblaze can be parsed by the assembly parser.

# TYPE B:   OPCODE RD    RA    IMMEDIATE
#           000000 00000 00000 0000000000000000

# CHECK:    addi
# BINARY:   001000 00001 00010 0000000000001111
# CHECK:    encoding: [0x20,0x22,0x00,0x0f]
            addi    r1, r2, 0x000F

# CHECK:    addic
# BINARY:   001010 00001 00010 0000000000001111
# CHECK:    encoding: [0x28,0x22,0x00,0x0f]
            addic   r1, r2, 0x000F

# CHECK:    addik
# BINARY:   001100 00001 00010 0000000000001111
# CHECK:    encoding: [0x30,0x22,0x00,0x0f]
            addik   r1, r2, 0x000F

# CHECK:    addikc
# BINARY:   001110 00001 00010 0000000000001111
# CHECK:    encoding: [0x38,0x22,0x00,0x0f]
            addikc  r1, r2, 0x000F

# CHECK:    andi
# BINARY:   101001 00001 00010 0000000000001111
# CHECK:    encoding: [0xa4,0x22,0x00,0x0f]
            andi    r1, r2, 0x000F

# CHECK:    andni
# BINARY:   101011 00001 00010 0000000000001111
# CHECK:    encoding: [0xac,0x22,0x00,0x0f]
            andni   r1, r2, 0x000F

# CHECK:    muli
# BINARY:   011000 00001 00010 0000000000001111
# CHECK:    encoding: [0x60,0x22,0x00,0x0f]
            muli    r1, r2, 0x000F

# CHECK:    ori
# BINARY:   101000 00001 00010 0000000000001111
# CHECK:    encoding: [0xa0,0x22,0x00,0x0f]
            ori     r1, r2, 0x000F

# CHECK:    rsubi
# BINARY:   001001 00001 00010 0000000000001111
# CHECK:    encoding: [0x24,0x22,0x00,0x0f]
            rsubi   r1, r2, 0x000F

# CHECK:    rsubic
# BINARY:   001011 00001 00010 0000000000001111
# CHECK:    encoding: [0x2c,0x22,0x00,0x0f]
            rsubic  r1, r2, 0x000F

# CHECK:    rsubik
# BINARY:   001101 00001 00010 0000000000001111
# CHECK:    encoding: [0x34,0x22,0x00,0x0f]
            rsubik  r1, r2, 0x000F

# CHECK:    rsubikc
# BINARY:   001111 00001 00010 0000000000001111
# CHECK:    encoding: [0x3c,0x22,0x00,0x0f]
            rsubikc r1, r2, 0x000F

# CHECK:    rtbd
# BINARY:   101101 10010 01111 0000000000001111
# CHECK:    encoding: [0xb6,0x4f,0x00,0x0f]
            rtbd r15, 0x000F

# CHECK:    rted
# BINARY:   101101 10001 01111 0000000000001111
# CHECK:    encoding: [0xb6,0x8f,0x00,0x0f]
            rted r15, 0x000F

# CHECK:    rtid
# BINARY:   101101 10001 01111 0000000000001111
# CHECK:    encoding: [0xb6,0x2f,0x00,0x0f]
            rtid r15, 0x000F

# CHECK:    rtsd
# BINARY:   101101 10000 01111 0000000000001111
# CHECK:    encoding: [0xb6,0x0f,0x00,0x0f]
            rtsd r15, 0x000F

# CHECK:    xori
# BINARY:   101010 00001 00010 0000000000001111
# CHECK:    encoding: [0xa8,0x22,0x00,0x0f]
            xori r1, r2, 0x000F
