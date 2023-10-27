# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        bclri.b         $w21, $w30, 2           # encoding: [0x79,0xf2,0xf5,0x49]
# CHECK:        bclri.h         $w24, $w21, 0           # encoding: [0x79,0xe0,0xae,0x09]
# CHECK:        bclri.w         $w23, $w30, 3           # encoding: [0x79,0xc3,0xf5,0xc9]
# CHECK:        bclri.d         $w9, $w11, 0            # encoding: [0x79,0x80,0x5a,0x49]
# CHECK:        binsli.b        $w25, $w12, 1           # encoding: [0x7b,0x71,0x66,0x49]
# CHECK:        binsli.h        $w21, $w22, 0           # encoding: [0x7b,0x60,0xb5,0x49]
# CHECK:        binsli.w        $w22, $w4, 0            # encoding: [0x7b,0x40,0x25,0x89]
# CHECK:        binsli.d        $w6, $w2, 6             # encoding: [0x7b,0x06,0x11,0x89]
# CHECK:        binsri.b        $w15, $w19, 0           # encoding: [0x7b,0xf0,0x9b,0xc9]
# CHECK:        binsri.h        $w8, $w30, 1            # encoding: [0x7b,0xe1,0xf2,0x09]
# CHECK:        binsri.w        $w2, $w19, 5            # encoding: [0x7b,0xc5,0x98,0x89]
# CHECK:        binsri.d        $w18, $w20, 1           # encoding: [0x7b,0x81,0xa4,0x89]
# CHECK:        bnegi.b         $w24, $w19, 0           # encoding: [0x7a,0xf0,0x9e,0x09]
# CHECK:        bnegi.h         $w28, $w11, 3           # encoding: [0x7a,0xe3,0x5f,0x09]
# CHECK:        bnegi.w         $w1, $w27, 5            # encoding: [0x7a,0xc5,0xd8,0x49]
# CHECK:        bnegi.d         $w4, $w21, 1            # encoding: [0x7a,0x81,0xa9,0x09]
# CHECK:        bseti.b         $w18, $w8, 0            # encoding: [0x7a,0x70,0x44,0x89]
# CHECK:        bseti.h         $w24, $w14, 2           # encoding: [0x7a,0x62,0x76,0x09]
# CHECK:        bseti.w         $w9, $w18, 4            # encoding: [0x7a,0x44,0x92,0x49]
# CHECK:        bseti.d         $w7, $w15, 1            # encoding: [0x7a,0x01,0x79,0xc9]
# CHECK:        sat_s.b         $w31, $w31, 2           # encoding: [0x78,0x72,0xff,0xca]
# CHECK:        sat_s.h         $w19, $w19, 0           # encoding: [0x78,0x60,0x9c,0xca]
# CHECK:        sat_s.w         $w19, $w29, 0           # encoding: [0x78,0x40,0xec,0xca]
# CHECK:        sat_s.d         $w11, $w22, 0           # encoding: [0x78,0x00,0xb2,0xca]
# CHECK:        sat_u.b         $w1, $w13, 3            # encoding: [0x78,0xf3,0x68,0x4a]
# CHECK:        sat_u.h         $w30, $w24, 4           # encoding: [0x78,0xe4,0xc7,0x8a]
# CHECK:        sat_u.w         $w31, $w13, 0           # encoding: [0x78,0xc0,0x6f,0xca]
# CHECK:        sat_u.d         $w29, $w16, 5           # encoding: [0x78,0x85,0x87,0x4a]
# CHECK:        slli.b          $w23, $w10, 1           # encoding: [0x78,0x71,0x55,0xc9]
# CHECK:        slli.h          $w9, $w18, 1            # encoding: [0x78,0x61,0x92,0x49]
# CHECK:        slli.w          $w11, $w29, 4           # encoding: [0x78,0x44,0xea,0xc9]
# CHECK:        slli.d          $w25, $w20, 1           # encoding: [0x78,0x01,0xa6,0x49]
# CHECK:        srai.b          $w24, $w29, 1           # encoding: [0x78,0xf1,0xee,0x09]
# CHECK:        srai.h          $w1, $w6, 0             # encoding: [0x78,0xe0,0x30,0x49]
# CHECK:        srai.w          $w7, $w26, 1            # encoding: [0x78,0xc1,0xd1,0xc9]
# CHECK:        srai.d          $w20, $w25, 3           # encoding: [0x78,0x83,0xcd,0x09]
# CHECK:        srari.b         $w5, $w25, 0            # encoding: [0x79,0x70,0xc9,0x4a]
# CHECK:        srari.h         $w7, $w6, 4             # encoding: [0x79,0x64,0x31,0xca]
# CHECK:        srari.w         $w17, $w11, 5           # encoding: [0x79,0x45,0x5c,0x4a]
# CHECK:        srari.d         $w21, $w25, 5           # encoding: [0x79,0x05,0xcd,0x4a]
# CHECK:        srli.b          $w2, $w0, 2             # encoding: [0x79,0x72,0x00,0x89]
# CHECK:        srli.h          $w31, $w31, 2           # encoding: [0x79,0x62,0xff,0xc9]
# CHECK:        srli.w          $w5, $w9, 4             # encoding: [0x79,0x44,0x49,0x49]
# CHECK:        srli.d          $w27, $w26, 5           # encoding: [0x79,0x05,0xd6,0xc9]
# CHECK:        srlri.b         $w18, $w3, 0            # encoding: [0x79,0xf0,0x1c,0x8a]
# CHECK:        srlri.h         $w1, $w2, 3             # encoding: [0x79,0xe3,0x10,0x4a]
# CHECK:        srlri.w         $w11, $w22, 2           # encoding: [0x79,0xc2,0xb2,0xca]
# CHECK:        srlri.d         $w24, $w10, 6           # encoding: [0x79,0x86,0x56,0x0a]

                bclri.b         $w21, $w30, 2
                bclri.h         $w24, $w21, 0
                bclri.w         $w23, $w30, 3
                bclri.d         $w9, $w11, 0
                binsli.b        $w25, $w12, 1
                binsli.h        $w21, $w22, 0
                binsli.w        $w22, $w4, 0
                binsli.d        $w6, $w2, 6
                binsri.b        $w15, $w19, 0
                binsri.h        $w8, $w30, 1
                binsri.w        $w2, $w19, 5
                binsri.d        $w18, $w20, 1
                bnegi.b         $w24, $w19, 0
                bnegi.h         $w28, $w11, 3
                bnegi.w         $w1, $w27, 5
                bnegi.d         $w4, $w21, 1
                bseti.b         $w18, $w8, 0
                bseti.h         $w24, $w14, 2
                bseti.w         $w9, $w18, 4
                bseti.d         $w7, $w15, 1
                sat_s.b         $w31, $w31, 2
                sat_s.h         $w19, $w19, 0
                sat_s.w         $w19, $w29, 0
                sat_s.d         $w11, $w22, 0
                sat_u.b         $w1, $w13, 3
                sat_u.h         $w30, $w24, 4
                sat_u.w         $w31, $w13, 0
                sat_u.d         $w29, $w16, 5
                slli.b          $w23, $w10, 1
                slli.h          $w9, $w18, 1
                slli.w          $w11, $w29, 4
                slli.d          $w25, $w20, 1
                srai.b          $w24, $w29, 1
                srai.h          $w1, $w6, 0
                srai.w          $w7, $w26, 1
                srai.d          $w20, $w25, 3
                srari.b         $w5, $w25, 0
                srari.h         $w7, $w6, 4
                srari.w         $w17, $w11, 5
                srari.d         $w21, $w25, 5
                srli.b          $w2, $w0, 2
                srli.h          $w31, $w31, 2
                srli.w          $w5, $w9, 4
                srli.d          $w27, $w26, 5
                srlri.b         $w18, $w3, 0
                srlri.h         $w1, $w2, 3
                srlri.w         $w11, $w22, 2
                srlri.d         $w24, $w10, 6
