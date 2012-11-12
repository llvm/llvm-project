# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to ensure that all special instructions and special registers can be
# parsed by the assembly parser correctly.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   011011 00000 00000 00000 00000000000

# CHECK:    mfs
# BINARY:   100101 00000 00000 10000 00000000000
# CHECK:    encoding: [0x94,0x00,0x80,0x00]
            mfs         r0, rpc

# CHECK:    msrclr
# BINARY:   100101 00000 100010 000000000000000
# CHECK:    encoding: [0x94,0x11,0x00,0x00]
            msrclr      r0, 0x0

# CHECK:    msrset
# BINARY:   100101 00000 100000 000000000000000
# CHECK:    encoding: [0x94,0x10,0x00,0x00]
            msrset      r0, 0x0

# CHECK:    mts
# BINARY:   100101 00000 00000 11 00000000000000
# CHECK:    encoding: [0x94,0x00,0xc0,0x00]
            mts         rpc, r0

# CHECK:    wdc
# BINARY:   100100 00000 00000 00001 00001100100
# CHECK:    encoding: [0x90,0x00,0x08,0x64]
            wdc         r0, r1

# CHECK:    wdc.clear
# BINARY:   100100 00000 00000 00001 00001100110
# CHECK:    encoding: [0x90,0x00,0x08,0x66]
            wdc.clear   r0, r1

# CHECK:    wdc.flush
# BINARY:   100100 00000 00000 00001 00001110100
# CHECK:    encoding: [0x90,0x00,0x08,0x74]
            wdc.flush   r0, r1

# CHECK:    wic
# BINARY:   100100 00000 00000 00001 00001101000
# CHECK:    encoding: [0x90,0x00,0x08,0x68]
            wic         r0, r1

# CHECK:    mfs
# BINARY:   100101 00001 00000 10000 00000000000
# CHECK:    encoding: [0x94,0x20,0x80,0x00]
            mfs         r1, rpc

# CHECK:    mfs
# BINARY:   100101 00001 00000 10000 00000000001
# CHECK:    encoding: [0x94,0x20,0x80,0x01]
            mfs         r1, rmsr

# CHECK:    mfs
# BINARY:   100101 00001 00000 10000 00000000011
# CHECK:    encoding: [0x94,0x20,0x80,0x03]
            mfs         r1, rear

# CHECK:    mfs
# BINARY:   100101 00001 00000 10000 00000000101
# CHECK:    encoding: [0x94,0x20,0x80,0x05]
            mfs         r1, resr

# CHECK:    mfs
# BINARY:   100101 00001 00000 10000 00000000111
# CHECK:    encoding: [0x94,0x20,0x80,0x07]
            mfs         r1, rfsr

# CHECK:    mfs
# BINARY:   100101 00001 00000 10000 00000001011
# CHECK:    encoding: [0x94,0x20,0x80,0x0b]
            mfs         r1, rbtr

# CHECK:    mfs
# BINARY:   100101 00001 00000 10000 00000001101
# CHECK:    encoding: [0x94,0x20,0x80,0x0d]
            mfs         r1, redr

# CHECK:    mfs
# BINARY:   100101 00001 00000 10010 00000000000
# CHECK:    encoding: [0x94,0x20,0x90,0x00]
            mfs         r1, rpid

# CHECK:    mfs
# BINARY:   100101 00001 00000 10010 00000000001
# CHECK:    encoding: [0x94,0x20,0x90,0x01]
            mfs         r1, rzpr

# CHECK:    mfs
# BINARY:   100101 00001 00000 10010 00000000010
# CHECK:    encoding: [0x94,0x20,0x90,0x02]
            mfs         r1, rtlbx

# CHECK:    mfs
# BINARY:   100101 00001 00000 10010 00000000100
# CHECK:    encoding: [0x94,0x20,0x90,0x04]
            mfs         r1, rtlbhi

# CHECK:    mfs
# BINARY:   100101 00001 00000 10010 00000000011
# CHECK:    encoding: [0x94,0x20,0x90,0x03]
            mfs         r1, rtlblo

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000000000
# CHECK:    encoding: [0x94,0x20,0xa0,0x00]
            mfs         r1, rpvr0

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000000001
# CHECK:    encoding: [0x94,0x20,0xa0,0x01]
            mfs         r1, rpvr1

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000000010
# CHECK:    encoding: [0x94,0x20,0xa0,0x02]
            mfs         r1, rpvr2

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000000011
# CHECK:    encoding: [0x94,0x20,0xa0,0x03]
            mfs         r1, rpvr3

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000000100
# CHECK:    encoding: [0x94,0x20,0xa0,0x04]
            mfs         r1, rpvr4

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000000101
# CHECK:    encoding: [0x94,0x20,0xa0,0x05]
            mfs         r1, rpvr5

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000000110
# CHECK:    encoding: [0x94,0x20,0xa0,0x06]
            mfs         r1, rpvr6

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000000111
# CHECK:    encoding: [0x94,0x20,0xa0,0x07]
            mfs         r1, rpvr7

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000001000
# CHECK:    encoding: [0x94,0x20,0xa0,0x08]
            mfs         r1, rpvr8

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000001001
# CHECK:    encoding: [0x94,0x20,0xa0,0x09]
            mfs         r1, rpvr9

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000001010
# CHECK:    encoding: [0x94,0x20,0xa0,0x0a]
            mfs         r1, rpvr10

# CHECK:    mfs
# BINARY:   100101 00001 00000 10100 00000001011
# CHECK:    encoding: [0x94,0x20,0xa0,0x0b]
            mfs         r1, rpvr11
