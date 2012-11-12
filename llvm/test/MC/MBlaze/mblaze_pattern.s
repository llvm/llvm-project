# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to ensure that all FPU instructions can be parsed by the
# assembly parser correctly.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   011011 00000 00000 00000 00000000000

# CHECK:    pcmpbf
# BINARY:   100000 00000 00001 00010 10000000000
# CHECK:    encoding: [0x80,0x01,0x14,0x00]
            pcmpbf      r0, r1, r2

# CHECK:    pcmpne
# BINARY:   100011 00000 00001 00010 10000000000
# CHECK:    encoding: [0x8c,0x01,0x14,0x00]
            pcmpne      r0, r1, r2

# CHECK:    pcmpeq
# BINARY:   100010 00000 00001 00010 10000000000
# CHECK:    encoding: [0x88,0x01,0x14,0x00]
            pcmpeq      r0, r1, r2
