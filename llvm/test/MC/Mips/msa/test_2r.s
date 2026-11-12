# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        fill.b  $w30, $9                # encoding: [0x7b,0x00,0x4f,0x9e]
# CHECK:        fill.h  $w31, $23               # encoding: [0x7b,0x01,0xbf,0xde]
# CHECK:        fill.w  $w16, $24               # encoding: [0x7b,0x02,0xc4,0x1e]
# CHECK:        nloc.b  $w21, $w0               # encoding: [0x7b,0x08,0x05,0x5e]
# CHECK:        nloc.h  $w18, $w31              # encoding: [0x7b,0x09,0xfc,0x9e]
# CHECK:        nloc.w  $w2, $w23               # encoding: [0x7b,0x0a,0xb8,0x9e]
# CHECK:        nloc.d  $w4, $w10               # encoding: [0x7b,0x0b,0x51,0x1e]
# CHECK:        nlzc.b  $w31, $w2               # encoding: [0x7b,0x0c,0x17,0xde]
# CHECK:        nlzc.h  $w27, $w22              # encoding: [0x7b,0x0d,0xb6,0xde]
# CHECK:        nlzc.w  $w10, $w29              # encoding: [0x7b,0x0e,0xea,0x9e]
# CHECK:        nlzc.d  $w25, $w9               # encoding: [0x7b,0x0f,0x4e,0x5e]
# CHECK:        pcnt.b  $w20, $w18              # encoding: [0x7b,0x04,0x95,0x1e]
# CHECK:        pcnt.h  $w0, $w8                # encoding: [0x7b,0x05,0x40,0x1e]
# CHECK:        pcnt.w  $w23, $w9               # encoding: [0x7b,0x06,0x4d,0xde]
# CHECK:        pcnt.d  $w21, $w24              # encoding: [0x7b,0x07,0xc5,0x5e]

                fill.b  $w30, $9
                fill.h  $w31, $23
                fill.w  $w16, $24
                nloc.b  $w21, $w0
                nloc.h  $w18, $w31
                nloc.w  $w2, $w23
                nloc.d  $w4, $w10
                nlzc.b  $w31, $w2
                nlzc.h  $w27, $w22
                nlzc.w  $w10, $w29
                nlzc.d  $w25, $w9
                pcnt.b  $w20, $w18
                pcnt.h  $w0, $w8
                pcnt.w  $w23, $w9
                pcnt.d  $w21, $w24
