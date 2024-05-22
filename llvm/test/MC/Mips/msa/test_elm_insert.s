# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        insert.b        $w23[3], $sp            # encoding: [0x79,0x03,0xed,0xd9]
# CHECK:        insert.h        $w20[2], $5             # encoding: [0x79,0x22,0x2d,0x19]
# CHECK:        insert.w        $w8[2], $15             # encoding: [0x79,0x32,0x7a,0x19]

                insert.b        $w23[3], $sp
                insert.h        $w20[2], $5
                insert.w        $w8[2], $15
