# RUN: llvm-mc -triple mblaze-unknown-unknown -show-encoding %s | FileCheck %s

# Test to ensure that all FPU instructions can be parsed by the
# assembly parser correctly.

# TYPE A:   OPCODE RD    RA    RB    FLAGS
# BINARY:   011011 00000 00000 00000 00000000000

# CHECK:    fadd
# BINARY:   010110 00000 00001 00010 00000000000
# CHECK:    encoding: [0x58,0x01,0x10,0x00]
            fadd         r0, r1, r2

# CHECK:    frsub
# BINARY:   010110 00000 00001 00010 00010000000
# CHECK:    encoding: [0x58,0x01,0x10,0x80]
            frsub        r0, r1, r2

# CHECK:    fmul
# BINARY:   010110 00000 00001 00010 00100000000
# CHECK:    encoding: [0x58,0x01,0x11,0x00]
            fmul         r0, r1, r2

# CHECK:    fdiv
# BINARY:   010110 00000 00001 00010 00110000000
# CHECK:    encoding: [0x58,0x01,0x11,0x80]
            fdiv         r0, r1, r2

# CHECK:    fsqrt
# BINARY:   010110 00000 00001 00000 01110000000
# CHECK:    encoding: [0x58,0x01,0x03,0x80]
            fsqrt        r0, r1

# CHECK:    fint
# BINARY:   010110 00000 00001 00000 01100000000
# CHECK:    encoding: [0x58,0x01,0x03,0x00]
            fint         r0, r1

# CHECK:    flt
# BINARY:   010110 00000 00001 00000 01010000000
# CHECK:    encoding: [0x58,0x01,0x02,0x80]
            flt          r0, r1

# CHECK:    fcmp.un
# BINARY:   010110 00000 00001 00010 01000000000
# CHECK:    encoding: [0x58,0x01,0x12,0x00]
            fcmp.un     r0, r1, r2

# CHECK:    fcmp.lt
# BINARY:   010110 00000 00001 00010 01000010000
# CHECK:    encoding: [0x58,0x01,0x12,0x10]
            fcmp.lt     r0, r1, r2

# CHECK:    fcmp.eq
# BINARY:   010110 00000 00001 00010 01000100000
# CHECK:    encoding: [0x58,0x01,0x12,0x20]
            fcmp.eq     r0, r1, r2

# CHECK:    fcmp.le
# BINARY:   010110 00000 00001 00010 01000110000
# CHECK:    encoding: [0x58,0x01,0x12,0x30]
            fcmp.le     r0, r1, r2

# CHECK:    fcmp.gt
# BINARY:   010110 00000 00001 00010 01001000000
# CHECK:    encoding: [0x58,0x01,0x12,0x40]
            fcmp.gt     r0, r1, r2

# CHECK:    fcmp.ne
# BINARY:   010110 00000 00001 00010 01001010000
# CHECK:    encoding: [0x58,0x01,0x12,0x50]
            fcmp.ne     r0, r1, r2

# CHECK:    fcmp.ge
# BINARY:   010110 00000 00001 00010 01001100000
# CHECK:    encoding: [0x58,0x01,0x12,0x60]
            fcmp.ge     r0, r1, r2
