# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        and.v   $w25, $w20, $w27        # encoding: [0x78,0x1b,0xa6,0x5e]
# CHECK:        bmnz.v  $w17, $w6, $w7          # encoding: [0x78,0x87,0x34,0x5e]
# CHECK:        bmz.v   $w3, $w17, $w9          # encoding: [0x78,0xa9,0x88,0xde]
# CHECK:        bsel.v  $w8, $w0, $w14          # encoding: [0x78,0xce,0x02,0x1e]
# CHECK:        nor.v   $w7, $w31, $w0          # encoding: [0x78,0x40,0xf9,0xde]
# CHECK:        or.v    $w24, $w26, $w30        # encoding: [0x78,0x3e,0xd6,0x1e]
# CHECK:        xor.v   $w7, $w27, $w15         # encoding: [0x78,0x6f,0xd9,0xde]

                and.v   $w25, $w20, $w27
                bmnz.v  $w17, $w6, $w7
                bmz.v   $w3, $w17, $w9
                bsel.v  $w8, $w0, $w14
                nor.v   $w7, $w31, $w0
                or.v    $w24, $w26, $w30
                xor.v   $w7, $w27, $w15
