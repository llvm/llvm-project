# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        ldi.b   $w8, 198                # encoding: [0x7b,0x06,0x32,0x07]
# CHECK:        ldi.h   $w20, 313               # encoding: [0x7b,0x29,0xcd,0x07]
# CHECK:        ldi.w   $w24, 492               # encoding: [0x7b,0x4f,0x66,0x07]
# CHECK:        ldi.d   $w27, -180              # encoding: [0x7b,0x7a,0x66,0xc7]

                ldi.b   $w8, 198
                ldi.h   $w20, 313
                ldi.w   $w24, 492
                ldi.d   $w27, -180
