# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        addvi.b         $w3, $w31, 30           # encoding: [0x78,0x1e,0xf8,0xc6]
# CHECK:        addvi.h         $w24, $w13, 26          # encoding: [0x78,0x3a,0x6e,0x06]
# CHECK:        addvi.w         $w26, $w20, 26          # encoding: [0x78,0x5a,0xa6,0x86]
# CHECK:        addvi.d         $w16, $w1, 21           # encoding: [0x78,0x75,0x0c,0x06]
# CHECK:        ceqi.b          $w24, $w21, -8          # encoding: [0x78,0x18,0xae,0x07]
# CHECK:        ceqi.h          $w31, $w15, 2           # encoding: [0x78,0x22,0x7f,0xc7]
# CHECK:        ceqi.w          $w12, $w1, -1           # encoding: [0x78,0x5f,0x0b,0x07]
# CHECK:        ceqi.d          $w24, $w22, 7           # encoding: [0x78,0x67,0xb6,0x07]
# CHECK:        clei_s.b        $w12, $w16, 1           # encoding: [0x7a,0x01,0x83,0x07]
# CHECK:        clei_s.h        $w2, $w10, -9           # encoding: [0x7a,0x37,0x50,0x87]
# CHECK:        clei_s.w        $w4, $w11, -10          # encoding: [0x7a,0x56,0x59,0x07]
# CHECK:        clei_s.d        $w0, $w29, -10          # encoding: [0x7a,0x76,0xe8,0x07]
# CHECK:        clei_u.b        $w21, $w17, 3           # encoding: [0x7a,0x83,0x8d,0x47]
# CHECK:        clei_u.h        $w29, $w7, 17           # encoding: [0x7a,0xb1,0x3f,0x47]
# CHECK:        clei_u.w        $w1, $w1, 2             # encoding: [0x7a,0xc2,0x08,0x47]
# CHECK:        clei_u.d        $w27, $w27, 29          # encoding: [0x7a,0xfd,0xde,0xc7]
# CHECK:        clti_s.b        $w19, $w13, -7          # encoding: [0x79,0x19,0x6c,0xc7]
# CHECK:        clti_s.h        $w15, $w10, -12         # encoding: [0x79,0x34,0x53,0xc7]
# CHECK:        clti_s.w        $w12, $w12, 11          # encoding: [0x79,0x4b,0x63,0x07]
# CHECK:        clti_s.d        $w29, $w20, -15         # encoding: [0x79,0x71,0xa7,0x47]
# CHECK:        clti_u.b        $w14, $w9, 29           # encoding: [0x79,0x9d,0x4b,0x87]
# CHECK:        clti_u.h        $w24, $w25, 25          # encoding: [0x79,0xb9,0xce,0x07]
# CHECK:        clti_u.w        $w1, $w1, 22            # encoding: [0x79,0xd6,0x08,0x47]
# CHECK:        clti_u.d        $w21, $w25, 1           # encoding: [0x79,0xe1,0xcd,0x47]
# CHECK:        maxi_s.b        $w22, $w21, 1           # encoding: [0x79,0x01,0xad,0x86]
# CHECK:        maxi_s.h        $w29, $w5, -8           # encoding: [0x79,0x38,0x2f,0x46]
# CHECK:        maxi_s.w        $w1, $w10, -12          # encoding: [0x79,0x54,0x50,0x46]
# CHECK:        maxi_s.d        $w13, $w29, -16         # encoding: [0x79,0x70,0xeb,0x46]
# CHECK:        maxi_u.b        $w20, $w0, 12           # encoding: [0x79,0x8c,0x05,0x06]
# CHECK:        maxi_u.h        $w1, $w14, 3            # encoding: [0x79,0xa3,0x70,0x46]
# CHECK:        maxi_u.w        $w27, $w22, 11          # encoding: [0x79,0xcb,0xb6,0xc6]
# CHECK:        maxi_u.d        $w26, $w6, 4            # encoding: [0x79,0xe4,0x36,0x86]
# CHECK:        mini_s.b        $w4, $w1, 1             # encoding: [0x7a,0x01,0x09,0x06]
# CHECK:        mini_s.h        $w27, $w27, -9          # encoding: [0x7a,0x37,0xde,0xc6]
# CHECK:        mini_s.w        $w28, $w11, 9           # encoding: [0x7a,0x49,0x5f,0x06]
# CHECK:        mini_s.d        $w11, $w10, 10          # encoding: [0x7a,0x6a,0x52,0xc6]
# CHECK:        mini_u.b        $w18, $w23, 27          # encoding: [0x7a,0x9b,0xbc,0x86]
# CHECK:        mini_u.h        $w7, $w26, 18           # encoding: [0x7a,0xb2,0xd1,0xc6]
# CHECK:        mini_u.w        $w11, $w12, 26          # encoding: [0x7a,0xda,0x62,0xc6]
# CHECK:        mini_u.d        $w11, $w15, 2           # encoding: [0x7a,0xe2,0x7a,0xc6]
# CHECK:        subvi.b         $w24, $w20, 19          # encoding: [0x78,0x93,0xa6,0x06]
# CHECK:        subvi.h         $w11, $w19, 4           # encoding: [0x78,0xa4,0x9a,0xc6]
# CHECK:        subvi.w         $w12, $w10, 11          # encoding: [0x78,0xcb,0x53,0x06]
# CHECK:        subvi.d         $w19, $w16, 7           # encoding: [0x78,0xe7,0x84,0xc6]

                addvi.b         $w3, $w31, 30
                addvi.h         $w24, $w13, 26
                addvi.w         $w26, $w20, 26
                addvi.d         $w16, $w1, 21
                ceqi.b          $w24, $w21, -8
                ceqi.h          $w31, $w15, 2
                ceqi.w          $w12, $w1, -1
                ceqi.d          $w24, $w22, 7
                clei_s.b        $w12, $w16, 1
                clei_s.h        $w2, $w10, -9
                clei_s.w        $w4, $w11, -10
                clei_s.d        $w0, $w29, -10
                clei_u.b        $w21, $w17, 3
                clei_u.h        $w29, $w7, 17
                clei_u.w        $w1, $w1, 2
                clei_u.d        $w27, $w27, 29
                clti_s.b        $w19, $w13, -7
                clti_s.h        $w15, $w10, -12
                clti_s.w        $w12, $w12, 11
                clti_s.d        $w29, $w20, -15
                clti_u.b        $w14, $w9, 29
                clti_u.h        $w24, $w25, 25
                clti_u.w        $w1, $w1, 22
                clti_u.d        $w21, $w25, 1
                maxi_s.b        $w22, $w21, 1
                maxi_s.h        $w29, $w5, -8
                maxi_s.w        $w1, $w10, -12
                maxi_s.d        $w13, $w29, -16
                maxi_u.b        $w20, $w0, 12
                maxi_u.h        $w1, $w14, 3
                maxi_u.w        $w27, $w22, 11
                maxi_u.d        $w26, $w6, 4
                mini_s.b        $w4, $w1, 1
                mini_s.h        $w27, $w27, -9
                mini_s.w        $w28, $w11, 9
                mini_s.d        $w11, $w10, 10
                mini_u.b        $w18, $w23, 27
                mini_u.h        $w7, $w26, 18
                mini_u.w        $w11, $w12, 26
                mini_u.d        $w11, $w15, 2
                subvi.b         $w24, $w20, 19
                subvi.h         $w11, $w19, 4
                subvi.w         $w12, $w10, 11
                subvi.d         $w19, $w16, 7
