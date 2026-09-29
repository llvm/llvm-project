# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        fadd.w          $w28, $w19, $w28        # encoding: [0x78,0x1c,0x9f,0x1b]
# CHECK:        fadd.d          $w13, $w2, $w29         # encoding: [0x78,0x3d,0x13,0x5b]
# CHECK:        fcaf.w          $w14, $w11, $w25        # encoding: [0x78,0x19,0x5b,0x9a]
# CHECK:        fcaf.d          $w1, $w1, $w19          # encoding: [0x78,0x33,0x08,0x5a]
# CHECK:        fceq.w          $w1, $w23, $w16         # encoding: [0x78,0x90,0xb8,0x5a]
# CHECK:        fceq.d          $w0, $w8, $w16          # encoding: [0x78,0xb0,0x40,0x1a]
# CHECK:        fcle.w          $w16, $w9, $w24         # encoding: [0x79,0x98,0x4c,0x1a]
# CHECK:        fcle.d          $w27, $w14, $w1         # encoding: [0x79,0xa1,0x76,0xda]
# CHECK:        fclt.w          $w28, $w8, $w8          # encoding: [0x79,0x08,0x47,0x1a]
# CHECK:        fclt.d          $w30, $w25, $w11        # encoding: [0x79,0x2b,0xcf,0x9a]
# CHECK:        fcne.w          $w2, $w18, $w23         # encoding: [0x78,0xd7,0x90,0x9c]
# CHECK:        fcne.d          $w14, $w20, $w15        # encoding: [0x78,0xef,0xa3,0x9c]
# CHECK:        fcor.w          $w10, $w18, $w25        # encoding: [0x78,0x59,0x92,0x9c]
# CHECK:        fcor.d          $w17, $w25, $w11        # encoding: [0x78,0x6b,0xcc,0x5c]
# CHECK:        fcueq.w         $w14, $w2, $w21         # encoding: [0x78,0xd5,0x13,0x9a]
# CHECK:        fcueq.d         $w29, $w3, $w7          # encoding: [0x78,0xe7,0x1f,0x5a]
# CHECK:        fcule.w         $w17, $w5, $w3          # encoding: [0x79,0xc3,0x2c,0x5a]
# CHECK:        fcule.d         $w31, $w1, $w30         # encoding: [0x79,0xfe,0x0f,0xda]
# CHECK:        fcult.w         $w6, $w25, $w9          # encoding: [0x79,0x49,0xc9,0x9a]
# CHECK:        fcult.d         $w27, $w8, $w17         # encoding: [0x79,0x71,0x46,0xda]
# CHECK:        fcun.w          $w4, $w20, $w8          # encoding: [0x78,0x48,0xa1,0x1a]
# CHECK:        fcun.d          $w29, $w11, $w3         # encoding: [0x78,0x63,0x5f,0x5a]
# CHECK:        fcune.w         $w13, $w18, $w19        # encoding: [0x78,0x93,0x93,0x5c]
# CHECK:        fcune.d         $w16, $w26, $w21        # encoding: [0x78,0xb5,0xd4,0x1c]
# CHECK:        fdiv.w          $w13, $w24, $w2         # encoding: [0x78,0xc2,0xc3,0x5b]
# CHECK:        fdiv.d          $w19, $w4, $w25         # encoding: [0x78,0xf9,0x24,0xdb]
# CHECK:        fexdo.h         $w8, $w0, $w16          # encoding: [0x7a,0x10,0x02,0x1b]
# CHECK:        fexdo.w         $w0, $w13, $w27         # encoding: [0x7a,0x3b,0x68,0x1b]
# CHECK:        fexp2.w         $w17, $w0, $w3          # encoding: [0x79,0xc3,0x04,0x5b]
# CHECK:        fexp2.d         $w22, $w0, $w10         # encoding: [0x79,0xea,0x05,0x9b]
# CHECK:        fmadd.w         $w29, $w6, $w23         # encoding: [0x79,0x17,0x37,0x5b]
# CHECK:        fmadd.d         $w11, $w28, $w21        # encoding: [0x79,0x35,0xe2,0xdb]
# CHECK:        fmax.w          $w0, $w23, $w13         # encoding: [0x7b,0x8d,0xb8,0x1b]
# CHECK:        fmax.d          $w26, $w18, $w8         # encoding: [0x7b,0xa8,0x96,0x9b]
# CHECK:        fmax_a.w        $w10, $w16, $w10        # encoding: [0x7b,0xca,0x82,0x9b]
# CHECK:        fmax_a.d        $w30, $w9, $w22         # encoding: [0x7b,0xf6,0x4f,0x9b]
# CHECK:        fmin.w          $w24, $w1, $w30         # encoding: [0x7b,0x1e,0x0e,0x1b]
# CHECK:        fmin.d          $w27, $w27, $w10        # encoding: [0x7b,0x2a,0xde,0xdb]
# CHECK:        fmin_a.w        $w10, $w29, $w20        # encoding: [0x7b,0x54,0xea,0x9b]
# CHECK:        fmin_a.d        $w13, $w30, $w24        # encoding: [0x7b,0x78,0xf3,0x5b]
# CHECK:        fmsub.w         $w17, $w25, $w0         # encoding: [0x79,0x40,0xcc,0x5b]
# CHECK:        fmsub.d         $w8, $w18, $w16         # encoding: [0x79,0x70,0x92,0x1b]
# CHECK:        fmul.w          $w3, $w15, $w15         # encoding: [0x78,0x8f,0x78,0xdb]
# CHECK:        fmul.d          $w9, $w30, $w10         # encoding: [0x78,0xaa,0xf2,0x5b]
# CHECK:        fsaf.w          $w25, $w5, $w10         # encoding: [0x7a,0x0a,0x2e,0x5a]
# CHECK:        fsaf.d          $w25, $w3, $w29         # encoding: [0x7a,0x3d,0x1e,0x5a]
# CHECK:        fseq.w          $w11, $w17, $w13        # encoding: [0x7a,0x8d,0x8a,0xda]
# CHECK:        fseq.d          $w29, $w0, $w31         # encoding: [0x7a,0xbf,0x07,0x5a]
# CHECK:        fsle.w          $w30, $w31, $w31        # encoding: [0x7b,0x9f,0xff,0x9a]
# CHECK:        fsle.d          $w18, $w23, $w24        # encoding: [0x7b,0xb8,0xbc,0x9a]
# CHECK:        fslt.w          $w12, $w5, $w6          # encoding: [0x7b,0x06,0x2b,0x1a]
# CHECK:        fslt.d          $w16, $w26, $w21        # encoding: [0x7b,0x35,0xd4,0x1a]
# CHECK:        fsne.w          $w30, $w1, $w12         # encoding: [0x7a,0xcc,0x0f,0x9c]
# CHECK:        fsne.d          $w14, $w13, $w23        # encoding: [0x7a,0xf7,0x6b,0x9c]
# CHECK:        fsor.w          $w27, $w13, $w27        # encoding: [0x7a,0x5b,0x6e,0xdc]
# CHECK:        fsor.d          $w12, $w24, $w11        # encoding: [0x7a,0x6b,0xc3,0x1c]
# CHECK:        fsub.w          $w31, $w26, $w1         # encoding: [0x78,0x41,0xd7,0xdb]
# CHECK:        fsub.d          $w19, $w17, $w27        # encoding: [0x78,0x7b,0x8c,0xdb]
# CHECK:        fsueq.w         $w16, $w24, $w25        # encoding: [0x7a,0xd9,0xc4,0x1a]
# CHECK:        fsueq.d         $w18, $w14, $w14        # encoding: [0x7a,0xee,0x74,0x9a]
# CHECK:        fsule.w         $w23, $w30, $w13        # encoding: [0x7b,0xcd,0xf5,0xda]
# CHECK:        fsule.d         $w2, $w11, $w26         # encoding: [0x7b,0xfa,0x58,0x9a]
# CHECK:        fsult.w         $w11, $w26, $w22        # encoding: [0x7b,0x56,0xd2,0xda]
# CHECK:        fsult.d         $w6, $w23, $w30         # encoding: [0x7b,0x7e,0xb9,0x9a]
# CHECK:        fsun.w          $w3, $w18, $w28         # encoding: [0x7a,0x5c,0x90,0xda]
# CHECK:        fsun.d          $w18, $w11, $w19        # encoding: [0x7a,0x73,0x5c,0x9a]
# CHECK:        fsune.w         $w16, $w31, $w2         # encoding: [0x7a,0x82,0xfc,0x1c]
# CHECK:        fsune.d         $w3, $w26, $w17         # encoding: [0x7a,0xb1,0xd0,0xdc]
# CHECK:        ftq.h           $w16, $w4, $w24         # encoding: [0x7a,0x98,0x24,0x1b]
# CHECK:        ftq.w           $w5, $w5, $w25          # encoding: [0x7a,0xb9,0x29,0x5b]
# CHECK:        madd_q.h        $w16, $w20, $w10        # encoding: [0x79,0x4a,0xa4,0x1c]
# CHECK:        madd_q.w        $w28, $w2, $w9          # encoding: [0x79,0x69,0x17,0x1c]
# CHECK:        maddr_q.h       $w8, $w18, $w9          # encoding: [0x7b,0x49,0x92,0x1c]
# CHECK:        maddr_q.w       $w29, $w12, $w16        # encoding: [0x7b,0x70,0x67,0x5c]
# CHECK:        msub_q.h        $w24, $w26, $w10        # encoding: [0x79,0x8a,0xd6,0x1c]
# CHECK:        msub_q.w        $w13, $w30, $w28        # encoding: [0x79,0xbc,0xf3,0x5c]
# CHECK:        msubr_q.h       $w12, $w21, $w11        # encoding: [0x7b,0x8b,0xab,0x1c]
# CHECK:        msubr_q.w       $w1, $w14, $w20         # encoding: [0x7b,0xb4,0x70,0x5c]
# CHECK:        mul_q.h         $w6, $w16, $w30         # encoding: [0x79,0x1e,0x81,0x9c]
# CHECK:        mul_q.w         $w16, $w1, $w4          # encoding: [0x79,0x24,0x0c,0x1c]
# CHECK:        mulr_q.h        $w6, $w20, $w19         # encoding: [0x7b,0x13,0xa1,0x9c]
# CHECK:        mulr_q.w        $w27, $w1, $w20         # encoding: [0x7b,0x34,0x0e,0xdc]

                fadd.w          $w28, $w19, $w28
                fadd.d          $w13, $w2, $w29
                fcaf.w          $w14, $w11, $w25
                fcaf.d          $w1, $w1, $w19
                fceq.w          $w1, $w23, $w16
                fceq.d          $w0, $w8, $w16
                fcle.w          $w16, $w9, $w24
                fcle.d          $w27, $w14, $w1
                fclt.w          $w28, $w8, $w8
                fclt.d          $w30, $w25, $w11
                fcne.w          $w2, $w18, $w23
                fcne.d          $w14, $w20, $w15
                fcor.w          $w10, $w18, $w25
                fcor.d          $w17, $w25, $w11
                fcueq.w         $w14, $w2, $w21
                fcueq.d         $w29, $w3, $w7
                fcule.w         $w17, $w5, $w3
                fcule.d         $w31, $w1, $w30
                fcult.w         $w6, $w25, $w9
                fcult.d         $w27, $w8, $w17
                fcun.w          $w4, $w20, $w8
                fcun.d          $w29, $w11, $w3
                fcune.w         $w13, $w18, $w19
                fcune.d         $w16, $w26, $w21
                fdiv.w          $w13, $w24, $w2
                fdiv.d          $w19, $w4, $w25
                fexdo.h         $w8, $w0, $w16
                fexdo.w         $w0, $w13, $w27
                fexp2.w         $w17, $w0, $w3
                fexp2.d         $w22, $w0, $w10
                fmadd.w         $w29, $w6, $w23
                fmadd.d         $w11, $w28, $w21
                fmax.w          $w0, $w23, $w13
                fmax.d          $w26, $w18, $w8
                fmax_a.w        $w10, $w16, $w10
                fmax_a.d        $w30, $w9, $w22
                fmin.w          $w24, $w1, $w30
                fmin.d          $w27, $w27, $w10
                fmin_a.w        $w10, $w29, $w20
                fmin_a.d        $w13, $w30, $w24
                fmsub.w         $w17, $w25, $w0
                fmsub.d         $w8, $w18, $w16
                fmul.w          $w3, $w15, $w15
                fmul.d          $w9, $w30, $w10
                fsaf.w          $w25, $w5, $w10
                fsaf.d          $w25, $w3, $w29
                fseq.w          $w11, $w17, $w13
                fseq.d          $w29, $w0, $w31
                fsle.w          $w30, $w31, $w31
                fsle.d          $w18, $w23, $w24
                fslt.w          $w12, $w5, $w6
                fslt.d          $w16, $w26, $w21
                fsne.w          $w30, $w1, $w12
                fsne.d          $w14, $w13, $w23
                fsor.w          $w27, $w13, $w27
                fsor.d          $w12, $w24, $w11
                fsub.w          $w31, $w26, $w1
                fsub.d          $w19, $w17, $w27
                fsueq.w         $w16, $w24, $w25
                fsueq.d         $w18, $w14, $w14
                fsule.w         $w23, $w30, $w13
                fsule.d         $w2, $w11, $w26
                fsult.w         $w11, $w26, $w22
                fsult.d         $w6, $w23, $w30
                fsun.w          $w3, $w18, $w28
                fsun.d          $w18, $w11, $w19
                fsune.w         $w16, $w31, $w2
                fsune.d         $w3, $w26, $w17
                ftq.h           $w16, $w4, $w24
                ftq.w           $w5, $w5, $w25
                madd_q.h        $w16, $w20, $w10
                madd_q.w        $w28, $w2, $w9
                maddr_q.h       $w8, $w18, $w9
                maddr_q.w       $w29, $w12, $w16
                msub_q.h        $w24, $w26, $w10
                msub_q.w        $w13, $w30, $w28
                msubr_q.h       $w12, $w21, $w11
                msubr_q.w       $w1, $w14, $w20
                mul_q.h         $w6, $w16, $w30
                mul_q.w         $w16, $w1, $w4
                mulr_q.h        $w6, $w20, $w19
                mulr_q.w        $w27, $w1, $w20
