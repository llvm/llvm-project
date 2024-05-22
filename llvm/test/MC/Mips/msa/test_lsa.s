# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        lsa        $8, $9, $10, 1              # encoding: [0x01,0x2a,0x40,0x05]
# CHECK:        lsa        $8, $9, $10, 2              # encoding: [0x01,0x2a,0x40,0x45]
# CHECK:        lsa        $8, $9, $10, 3              # encoding: [0x01,0x2a,0x40,0x85]
# CHECK:        lsa        $8, $9, $10, 4              # encoding: [0x01,0x2a,0x40,0xc5]

                lsa        $8, $9, $10, 1
                lsa        $8, $9, $10, 2
                lsa        $8, $9, $10, 3
                lsa        $8, $9, $10, 4
