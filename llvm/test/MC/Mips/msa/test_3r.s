# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        add_a.b         $w26, $w9, $w4                  # encoding: [0x78,0x04,0x4e,0x90]
# CHECK:        add_a.h         $w23, $w27, $w31                # encoding: [0x78,0x3f,0xdd,0xd0]
# CHECK:        add_a.w         $w11, $w6, $w22                 # encoding: [0x78,0x56,0x32,0xd0]
# CHECK:        add_a.d         $w6, $w10, $w0                  # encoding: [0x78,0x60,0x51,0x90]
# CHECK:        adds_a.b        $w19, $w24, $w19                # encoding: [0x78,0x93,0xc4,0xd0]
# CHECK:        adds_a.h        $w25, $w6, $w4                  # encoding: [0x78,0xa4,0x36,0x50]
# CHECK:        adds_a.w        $w25, $w17, $w27                # encoding: [0x78,0xdb,0x8e,0x50]
# CHECK:        adds_a.d        $w15, $w18, $w26                # encoding: [0x78,0xfa,0x93,0xd0]
# CHECK:        adds_s.b        $w29, $w11, $w19                # encoding: [0x79,0x13,0x5f,0x50]
# CHECK:        adds_s.h        $w5, $w23, $w26                 # encoding: [0x79,0x3a,0xb9,0x50]
# CHECK:        adds_s.w        $w16, $w14, $w13                # encoding: [0x79,0x4d,0x74,0x10]
# CHECK:        adds_s.d        $w2, $w14, $w28                 # encoding: [0x79,0x7c,0x70,0x90]
# CHECK:        adds_u.b        $w3, $w17, $w14                 # encoding: [0x79,0x8e,0x88,0xd0]
# CHECK:        adds_u.h        $w10, $w30, $w4                 # encoding: [0x79,0xa4,0xf2,0x90]
# CHECK:        adds_u.w        $w15, $w18, $w20                # encoding: [0x79,0xd4,0x93,0xd0]
# CHECK:        adds_u.d        $w30, $w10, $w9                 # encoding: [0x79,0xe9,0x57,0x90]
# CHECK:        addv.b          $w24, $w20, $w21                # encoding: [0x78,0x15,0xa6,0x0e]
# CHECK:        addv.h          $w4, $w13, $w27                 # encoding: [0x78,0x3b,0x69,0x0e]
# CHECK:        addv.w          $w19, $w11, $w14                # encoding: [0x78,0x4e,0x5c,0xce]
# CHECK:        addv.d          $w2, $w21, $w31                 # encoding: [0x78,0x7f,0xa8,0x8e]
# CHECK:        asub_s.b        $w23, $w16, $w3                 # encoding: [0x7a,0x03,0x85,0xd1]
# CHECK:        asub_s.h        $w22, $w17, $w25                # encoding: [0x7a,0x39,0x8d,0x91]
# CHECK:        asub_s.w        $w24, $w1, $w9                  # encoding: [0x7a,0x49,0x0e,0x11]
# CHECK:        asub_s.d        $w13, $w12, $w12                # encoding: [0x7a,0x6c,0x63,0x51]
# CHECK:        asub_u.b        $w10, $w29, $w11                # encoding: [0x7a,0x8b,0xea,0x91]
# CHECK:        asub_u.h        $w18, $w9, $w15                 # encoding: [0x7a,0xaf,0x4c,0x91]
# CHECK:        asub_u.w        $w10, $w19, $w31                # encoding: [0x7a,0xdf,0x9a,0x91]
# CHECK:        asub_u.d        $w17, $w10, $w0                 # encoding: [0x7a,0xe0,0x54,0x51]
# CHECK:        ave_s.b         $w2, $w5, $w1                   # encoding: [0x7a,0x01,0x28,0x90]
# CHECK:        ave_s.h         $w16, $w19, $w9                 # encoding: [0x7a,0x29,0x9c,0x10]
# CHECK:        ave_s.w         $w17, $w31, $w5                 # encoding: [0x7a,0x45,0xfc,0x50]
# CHECK:        ave_s.d         $w27, $w25, $w10                # encoding: [0x7a,0x6a,0xce,0xd0]
# CHECK:        ave_u.b         $w16, $w19, $w9                 # encoding: [0x7a,0x89,0x9c,0x10]
# CHECK:        ave_u.h         $w28, $w28, $w11                # encoding: [0x7a,0xab,0xe7,0x10]
# CHECK:        ave_u.w         $w11, $w12, $w11                # encoding: [0x7a,0xcb,0x62,0xd0]
# CHECK:        ave_u.d         $w30, $w19, $w28                # encoding: [0x7a,0xfc,0x9f,0x90]
# CHECK:        aver_s.b        $w26, $w16, $w2                 # encoding: [0x7b,0x02,0x86,0x90]
# CHECK:        aver_s.h        $w31, $w27, $w27                # encoding: [0x7b,0x3b,0xdf,0xd0]
# CHECK:        aver_s.w        $w28, $w18, $w25                # encoding: [0x7b,0x59,0x97,0x10]
# CHECK:        aver_s.d        $w29, $w21, $w27                # encoding: [0x7b,0x7b,0xaf,0x50]
# CHECK:        aver_u.b        $w29, $w26, $w3                 # encoding: [0x7b,0x83,0xd7,0x50]
# CHECK:        aver_u.h        $w18, $w18, $w9                 # encoding: [0x7b,0xa9,0x94,0x90]
# CHECK:        aver_u.w        $w17, $w25, $w29                # encoding: [0x7b,0xdd,0xcc,0x50]
# CHECK:        aver_u.d        $w22, $w22, $w19                # encoding: [0x7b,0xf3,0xb5,0x90]
# CHECK:        bclr.b          $w2, $w15, $w29                 # encoding: [0x79,0x9d,0x78,0x8d]
# CHECK:        bclr.h          $w16, $w21, $w28                # encoding: [0x79,0xbc,0xac,0x0d]
# CHECK:        bclr.w          $w19, $w2, $w9                  # encoding: [0x79,0xc9,0x14,0xcd]
# CHECK:        bclr.d          $w27, $w31, $w4                 # encoding: [0x79,0xe4,0xfe,0xcd]
# CHECK:        binsl.b         $w5, $w16, $w24                 # encoding: [0x7b,0x18,0x81,0x4d]
# CHECK:        binsl.h         $w30, $w5, $w10                 # encoding: [0x7b,0x2a,0x2f,0x8d]
# CHECK:        binsl.w         $w14, $w15, $w13                # encoding: [0x7b,0x4d,0x7b,0x8d]
# CHECK:        binsl.d         $w23, $w20, $w12                # encoding: [0x7b,0x6c,0xa5,0xcd]
# CHECK:        binsr.b         $w22, $w11, $w2                 # encoding: [0x7b,0x82,0x5d,0x8d]
# CHECK:        binsr.h         $w0, $w26, $w6                  # encoding: [0x7b,0xa6,0xd0,0x0d]
# CHECK:        binsr.w         $w26, $w3, $w28                 # encoding: [0x7b,0xdc,0x1e,0x8d]
# CHECK:        binsr.d         $w0, $w0, $w21                  # encoding: [0x7b,0xf5,0x00,0x0d]
# CHECK:        bneg.b          $w0, $w11, $w24                 # encoding: [0x7a,0x98,0x58,0x0d]
# CHECK:        bneg.h          $w28, $w16, $w4                 # encoding: [0x7a,0xa4,0x87,0x0d]
# CHECK:        bneg.w          $w3, $w26, $w19                 # encoding: [0x7a,0xd3,0xd0,0xcd]
# CHECK:        bneg.d          $w13, $w29, $w15                # encoding: [0x7a,0xef,0xeb,0x4d]
# CHECK:        bset.b          $w31, $w5, $w31                 # encoding: [0x7a,0x1f,0x2f,0xcd]
# CHECK:        bset.h          $w14, $w12, $w6                 # encoding: [0x7a,0x26,0x63,0x8d]
# CHECK:        bset.w          $w31, $w9, $w12                 # encoding: [0x7a,0x4c,0x4f,0xcd]
# CHECK:        bset.d          $w5, $w22, $w5                  # encoding: [0x7a,0x65,0xb1,0x4d]
# CHECK:        ceq.b           $w31, $w31, $w18                # encoding: [0x78,0x12,0xff,0xcf]
# CHECK:        ceq.h           $w10, $w27, $w9                 # encoding: [0x78,0x29,0xda,0x8f]
# CHECK:        ceq.w           $w9, $w5, $w14                  # encoding: [0x78,0x4e,0x2a,0x4f]
# CHECK:        ceq.d           $w5, $w17, $w0                  # encoding: [0x78,0x60,0x89,0x4f]
# CHECK:        cle_s.b         $w23, $w4, $w9                  # encoding: [0x7a,0x09,0x25,0xcf]
# CHECK:        cle_s.h         $w22, $w27, $w19                # encoding: [0x7a,0x33,0xdd,0x8f]
# CHECK:        cle_s.w         $w30, $w26, $w10                # encoding: [0x7a,0x4a,0xd7,0x8f]
# CHECK:        cle_s.d         $w18, $w5, $w10                 # encoding: [0x7a,0x6a,0x2c,0x8f]
# CHECK:        cle_u.b         $w1, $w25, $w0                  # encoding: [0x7a,0x80,0xc8,0x4f]
# CHECK:        cle_u.h         $w7, $w0, $w29                  # encoding: [0x7a,0xbd,0x01,0xcf]
# CHECK:        cle_u.w         $w25, $w18, $w1                 # encoding: [0x7a,0xc1,0x96,0x4f]
# CHECK:        cle_u.d         $w6, $w0, $w30                  # encoding: [0x7a,0xfe,0x01,0x8f]
# CHECK:        clt_s.b         $w25, $w2, $w21                 # encoding: [0x79,0x15,0x16,0x4f]
# CHECK:        clt_s.h         $w2, $w19, $w9                  # encoding: [0x79,0x29,0x98,0x8f]
# CHECK:        clt_s.w         $w23, $w8, $w16                 # encoding: [0x79,0x50,0x45,0xcf]
# CHECK:        clt_s.d         $w7, $w30, $w12                 # encoding: [0x79,0x6c,0xf1,0xcf]
# CHECK:        clt_u.b         $w2, $w31, $w13                 # encoding: [0x79,0x8d,0xf8,0x8f]
# CHECK:        clt_u.h         $w16, $w31, $w23                # encoding: [0x79,0xb7,0xfc,0x0f]
# CHECK:        clt_u.w         $w3, $w24, $w9                  # encoding: [0x79,0xc9,0xc0,0xcf]
# CHECK:        clt_u.d         $w7, $w0, $w1                   # encoding: [0x79,0xe1,0x01,0xcf]
# CHECK:        div_s.b         $w29, $w3, $w18                 # encoding: [0x7a,0x12,0x1f,0x52]
# CHECK:        div_s.h         $w17, $w16, $w13                # encoding: [0x7a,0x2d,0x84,0x52]
# CHECK:        div_s.w         $w4, $w25, $w30                 # encoding: [0x7a,0x5e,0xc9,0x12]
# CHECK:        div_s.d         $w31, $w9, $w20                 # encoding: [0x7a,0x74,0x4f,0xd2]
# CHECK:        div_u.b         $w6, $w29, $w10                 # encoding: [0x7a,0x8a,0xe9,0x92]
# CHECK:        div_u.h         $w24, $w21, $w14                # encoding: [0x7a,0xae,0xae,0x12]
# CHECK:        div_u.w         $w29, $w14, $w25                # encoding: [0x7a,0xd9,0x77,0x52]
# CHECK:        div_u.d         $w31, $w1, $w21                 # encoding: [0x7a,0xf5,0x0f,0xd2]
# CHECK:        dotp_s.h        $w23, $w22, $w25                # encoding: [0x78,0x39,0xb5,0xd3]
# CHECK:        dotp_s.w        $w20, $w14, $w5                 # encoding: [0x78,0x45,0x75,0x13]
# CHECK:        dotp_s.d        $w17, $w2, $w22                 # encoding: [0x78,0x76,0x14,0x53]
# CHECK:        dotp_u.h        $w13, $w2, $w6                  # encoding: [0x78,0xa6,0x13,0x53]
# CHECK:        dotp_u.w        $w15, $w22, $w21                # encoding: [0x78,0xd5,0xb3,0xd3]
# CHECK:        dotp_u.d        $w4, $w16, $w26                 # encoding: [0x78,0xfa,0x81,0x13]
# CHECK:        dpadd_s.h       $w1, $w28, $w22                 # encoding: [0x79,0x36,0xe0,0x53]
# CHECK:        dpadd_s.w       $w10, $w1, $w12                 # encoding: [0x79,0x4c,0x0a,0x93]
# CHECK:        dpadd_s.d       $w3, $w21, $w27                 # encoding: [0x79,0x7b,0xa8,0xd3]
# CHECK:        dpadd_u.h       $w17, $w5, $w20                 # encoding: [0x79,0xb4,0x2c,0x53]
# CHECK:        dpadd_u.w       $w24, $w8, $w16                 # encoding: [0x79,0xd0,0x46,0x13]
# CHECK:        dpadd_u.d       $w15, $w29, $w16                # encoding: [0x79,0xf0,0xeb,0xd3]
# CHECK:        dpsub_s.h       $w4, $w11, $w12                 # encoding: [0x7a,0x2c,0x59,0x13]
# CHECK:        dpsub_s.w       $w4, $w7, $w6                   # encoding: [0x7a,0x46,0x39,0x13]
# CHECK:        dpsub_s.d       $w31, $w12, $w28                # encoding: [0x7a,0x7c,0x67,0xd3]
# CHECK:        dpsub_u.h       $w4, $w25, $w17                 # encoding: [0x7a,0xb1,0xc9,0x13]
# CHECK:        dpsub_u.w       $w19, $w25, $w16                # encoding: [0x7a,0xd0,0xcc,0xd3]
# CHECK:        dpsub_u.d       $w7, $w10, $w26                 # encoding: [0x7a,0xfa,0x51,0xd3]
# CHECK:        hadd_s.h        $w28, $w24, $w2                 # encoding: [0x7a,0x22,0xc7,0x15]
# CHECK:        hadd_s.w        $w24, $w17, $w11                # encoding: [0x7a,0x4b,0x8e,0x15]
# CHECK:        hadd_s.d        $w17, $w15, $w20                # encoding: [0x7a,0x74,0x7c,0x55]
# CHECK:        hadd_u.h        $w12, $w29, $w17                # encoding: [0x7a,0xb1,0xeb,0x15]
# CHECK:        hadd_u.w        $w9, $w5, $w6                   # encoding: [0x7a,0xc6,0x2a,0x55]
# CHECK:        hadd_u.d        $w1, $w20, $w6                  # encoding: [0x7a,0xe6,0xa0,0x55]
# CHECK:        hsub_s.h        $w16, $w14, $w29                # encoding: [0x7b,0x3d,0x74,0x15]
# CHECK:        hsub_s.w        $w9, $w13, $w11                 # encoding: [0x7b,0x4b,0x6a,0x55]
# CHECK:        hsub_s.d        $w30, $w18, $w14                # encoding: [0x7b,0x6e,0x97,0x95]
# CHECK:        hsub_u.h        $w7, $w12, $w14                 # encoding: [0x7b,0xae,0x61,0xd5]
# CHECK:        hsub_u.w        $w21, $w5, $w5                  # encoding: [0x7b,0xc5,0x2d,0x55]
# CHECK:        hsub_u.d        $w11, $w12, $w31                # encoding: [0x7b,0xff,0x62,0xd5]
# CHECK:        ilvev.b         $w18, $w16, $w30                # encoding: [0x7b,0x1e,0x84,0x94]
# CHECK:        ilvev.h         $w14, $w0, $w13                 # encoding: [0x7b,0x2d,0x03,0x94]
# CHECK:        ilvev.w         $w12, $w25, $w22                # encoding: [0x7b,0x56,0xcb,0x14]
# CHECK:        ilvev.d         $w30, $w27, $w3                 # encoding: [0x7b,0x63,0xdf,0x94]
# CHECK:        ilvl.b          $w29, $w3, $w21                 # encoding: [0x7a,0x15,0x1f,0x54]
# CHECK:        ilvl.h          $w27, $w10, $w17                # encoding: [0x7a,0x31,0x56,0xd4]
# CHECK:        ilvl.w          $w6, $w1, $w0                   # encoding: [0x7a,0x40,0x09,0x94]
# CHECK:        ilvl.d          $w3, $w16, $w24                 # encoding: [0x7a,0x78,0x80,0xd4]
# CHECK:        ilvod.b         $w11, $w5, $w20                 # encoding: [0x7b,0x94,0x2a,0xd4]
# CHECK:        ilvod.h         $w18, $w13, $w31                # encoding: [0x7b,0xbf,0x6c,0x94]
# CHECK:        ilvod.w         $w29, $w16, $w24                # encoding: [0x7b,0xd8,0x87,0x54]
# CHECK:        ilvod.d         $w22, $w12, $w29                # encoding: [0x7b,0xfd,0x65,0x94]
# CHECK:        ilvr.b          $w4, $w30, $w6                  # encoding: [0x7a,0x86,0xf1,0x14]
# CHECK:        ilvr.h          $w28, $w19, $w29                # encoding: [0x7a,0xbd,0x9f,0x14]
# CHECK:        ilvr.w          $w18, $w20, $w21                # encoding: [0x7a,0xd5,0xa4,0x94]
# CHECK:        ilvr.d          $w23, $w30, $w12                # encoding: [0x7a,0xec,0xf5,0xd4]
# CHECK:        maddv.b         $w17, $w31, $w29                # encoding: [0x78,0x9d,0xfc,0x52]
# CHECK:        maddv.h         $w7, $w24, $w9                  # encoding: [0x78,0xa9,0xc1,0xd2]
# CHECK:        maddv.w         $w22, $w22, $w20                # encoding: [0x78,0xd4,0xb5,0x92]
# CHECK:        maddv.d         $w30, $w26, $w20                # encoding: [0x78,0xf4,0xd7,0x92]
# CHECK:        max_a.b         $w23, $w11, $w23                # encoding: [0x7b,0x17,0x5d,0xce]
# CHECK:        max_a.h         $w20, $w5, $w30                 # encoding: [0x7b,0x3e,0x2d,0x0e]
# CHECK:        max_a.w         $w7, $w18, $w30                 # encoding: [0x7b,0x5e,0x91,0xce]
# CHECK:        max_a.d         $w8, $w8, $w31                  # encoding: [0x7b,0x7f,0x42,0x0e]
# CHECK:        max_s.b         $w10, $w1, $w19                 # encoding: [0x79,0x13,0x0a,0x8e]
# CHECK:        max_s.h         $w15, $w29, $w17                # encoding: [0x79,0x31,0xeb,0xce]
# CHECK:        max_s.w         $w15, $w29, $w14                # encoding: [0x79,0x4e,0xeb,0xce]
# CHECK:        max_s.d         $w25, $w24, $w3                 # encoding: [0x79,0x63,0xc6,0x4e]
# CHECK:        max_u.b         $w12, $w24, $w5                 # encoding: [0x79,0x85,0xc3,0x0e]
# CHECK:        max_u.h         $w5, $w6, $w7                   # encoding: [0x79,0xa7,0x31,0x4e]
# CHECK:        max_u.w         $w16, $w4, $w7                  # encoding: [0x79,0xc7,0x24,0x0e]
# CHECK:        max_u.d         $w26, $w12, $w24                # encoding: [0x79,0xf8,0x66,0x8e]
# CHECK:        min_a.b         $w4, $w26, $w1                  # encoding: [0x7b,0x81,0xd1,0x0e]
# CHECK:        min_a.h         $w12, $w13, $w31                # encoding: [0x7b,0xbf,0x6b,0x0e]
# CHECK:        min_a.w         $w28, $w20, $w0                 # encoding: [0x7b,0xc0,0xa7,0x0e]
# CHECK:        min_a.d         $w12, $w20, $w19                # encoding: [0x7b,0xf3,0xa3,0x0e]
# CHECK:        min_s.b         $w19, $w3, $w14                 # encoding: [0x7a,0x0e,0x1c,0xce]
# CHECK:        min_s.h         $w27, $w21, $w8                 # encoding: [0x7a,0x28,0xae,0xce]
# CHECK:        min_s.w         $w0, $w14, $w30                 # encoding: [0x7a,0x5e,0x70,0x0e]
# CHECK:        min_s.d         $w6, $w8, $w21                  # encoding: [0x7a,0x75,0x41,0x8e]
# CHECK:        min_u.b         $w22, $w26, $w8                 # encoding: [0x7a,0x88,0xd5,0x8e]
# CHECK:        min_u.h         $w7, $w27, $w12                 # encoding: [0x7a,0xac,0xd9,0xce]
# CHECK:        min_u.w         $w8, $w20, $w14                 # encoding: [0x7a,0xce,0xa2,0x0e]
# CHECK:        min_u.d         $w26, $w14, $w15                # encoding: [0x7a,0xef,0x76,0x8e]
# CHECK:        mod_s.b         $w18, $w1, $w26                 # encoding: [0x7b,0x1a,0x0c,0x92]
# CHECK:        mod_s.h         $w31, $w30, $w28                # encoding: [0x7b,0x3c,0xf7,0xd2]
# CHECK:        mod_s.w         $w2, $w6, $w13                  # encoding: [0x7b,0x4d,0x30,0x92]
# CHECK:        mod_s.d         $w21, $w27, $w22                # encoding: [0x7b,0x76,0xdd,0x52]
# CHECK:        mod_u.b         $w16, $w7, $w13                 # encoding: [0x7b,0x8d,0x3c,0x12]
# CHECK:        mod_u.h         $w24, $w8, $w7                  # encoding: [0x7b,0xa7,0x46,0x12]
# CHECK:        mod_u.w         $w30, $w2, $w17                 # encoding: [0x7b,0xd1,0x17,0x92]
# CHECK:        mod_u.d         $w31, $w2, $w25                 # encoding: [0x7b,0xf9,0x17,0xd2]
# CHECK:        msubv.b         $w14, $w5, $w12                 # encoding: [0x79,0x0c,0x2b,0x92]
# CHECK:        msubv.h         $w6, $w7, $w30                  # encoding: [0x79,0x3e,0x39,0x92]
# CHECK:        msubv.w         $w13, $w2, $w21                 # encoding: [0x79,0x55,0x13,0x52]
# CHECK:        msubv.d         $w16, $w14, $w27                # encoding: [0x79,0x7b,0x74,0x12]
# CHECK:        mulv.b          $w20, $w3, $w13                 # encoding: [0x78,0x0d,0x1d,0x12]
# CHECK:        mulv.h          $w27, $w26, $w14                # encoding: [0x78,0x2e,0xd6,0xd2]
# CHECK:        mulv.w          $w10, $w29, $w3                 # encoding: [0x78,0x43,0xea,0x92]
# CHECK:        mulv.d          $w7, $w19, $w29                 # encoding: [0x78,0x7d,0x99,0xd2]
# CHECK:        pckev.b         $w5, $w27, $w7                  # encoding: [0x79,0x07,0xd9,0x54]
# CHECK:        pckev.h         $w1, $w4, $w27                  # encoding: [0x79,0x3b,0x20,0x54]
# CHECK:        pckev.w         $w30, $w20, $w0                 # encoding: [0x79,0x40,0xa7,0x94]
# CHECK:        pckev.d         $w6, $w1, $w15                  # encoding: [0x79,0x6f,0x09,0x94]
# CHECK:        pckod.b         $w18, $w28, $w30                # encoding: [0x79,0x9e,0xe4,0x94]
# CHECK:        pckod.h         $w26, $w5, $w8                  # encoding: [0x79,0xa8,0x2e,0x94]
# CHECK:        pckod.w         $w9, $w4, $w2                   # encoding: [0x79,0xc2,0x22,0x54]
# CHECK:        pckod.d         $w30, $w22, $w20                # encoding: [0x79,0xf4,0xb7,0x94]
# CHECK:        sld.b           $w5, $w23[$12]                  # encoding: [0x78,0x0c,0xb9,0x54]
# CHECK:        sld.h           $w1, $w23[$3]                   # encoding: [0x78,0x23,0xb8,0x54]
# CHECK:        sld.w           $w20, $w8[$9]                   # encoding: [0x78,0x49,0x45,0x14]
# CHECK:        sld.d           $w7, $w23[$fp]                  # encoding: [0x78,0x7e,0xb9,0xd4]
# CHECK:        sll.b           $w3, $w0, $w17                  # encoding: [0x78,0x11,0x00,0xcd]
# CHECK:        sll.h           $w17, $w27, $w3                 # encoding: [0x78,0x23,0xdc,0x4d]
# CHECK:        sll.w           $w16, $w7, $w6                  # encoding: [0x78,0x46,0x3c,0x0d]
# CHECK:        sll.d           $w9, $w0, $w26                  # encoding: [0x78,0x7a,0x02,0x4d]
# CHECK:        splat.b         $w28, $w1[$1]                   # encoding: [0x78,0x81,0x0f,0x14]
# CHECK:        splat.h         $w2, $w11[$11]                  # encoding: [0x78,0xab,0x58,0x94]
# CHECK:        splat.w         $w22, $w0[$11]                  # encoding: [0x78,0xcb,0x05,0x94]
# CHECK:        splat.d         $w0, $w0[$2]                    # encoding: [0x78,0xe2,0x00,0x14]
# CHECK:        sra.b           $w28, $w4, $w17                 # encoding: [0x78,0x91,0x27,0x0d]
# CHECK:        sra.h           $w13, $w9, $w3                  # encoding: [0x78,0xa3,0x4b,0x4d]
# CHECK:        sra.w           $w27, $w21, $w19                # encoding: [0x78,0xd3,0xae,0xcd]
# CHECK:        sra.d           $w30, $w8, $w23                 # encoding: [0x78,0xf7,0x47,0x8d]
# CHECK:        srar.b          $w19, $w18, $w18                # encoding: [0x78,0x92,0x94,0xd5]
# CHECK:        srar.h          $w7, $w23, $w8                  # encoding: [0x78,0xa8,0xb9,0xd5]
# CHECK:        srar.w          $w1, $w12, $w2                  # encoding: [0x78,0xc2,0x60,0x55]
# CHECK:        srar.d          $w21, $w7, $w14                 # encoding: [0x78,0xee,0x3d,0x55]
# CHECK:        srl.b           $w12, $w3, $w19                 # encoding: [0x79,0x13,0x1b,0x0d]
# CHECK:        srl.h           $w23, $w31, $w20                # encoding: [0x79,0x34,0xfd,0xcd]
# CHECK:        srl.w           $w18, $w27, $w11                # encoding: [0x79,0x4b,0xdc,0x8d]
# CHECK:        srl.d           $w3, $w12, $w26                 # encoding: [0x79,0x7a,0x60,0xcd]
# CHECK:        srlr.b          $w15, $w21, $w11                # encoding: [0x79,0x0b,0xab,0xd5]
# CHECK:        srlr.h          $w21, $w13, $w19                # encoding: [0x79,0x33,0x6d,0x55]
# CHECK:        srlr.w          $w6, $w30, $w3                  # encoding: [0x79,0x43,0xf1,0x95]
# CHECK:        srlr.d          $w1, $w2, $w14                  # encoding: [0x79,0x6e,0x10,0x55]
# CHECK:        subs_s.b        $w25, $w15, $w1                 # encoding: [0x78,0x01,0x7e,0x51]
# CHECK:        subs_s.h        $w28, $w25, $w22                # encoding: [0x78,0x36,0xcf,0x11]
# CHECK:        subs_s.w        $w10, $w12, $w21                # encoding: [0x78,0x55,0x62,0x91]
# CHECK:        subs_s.d        $w4, $w20, $w18                 # encoding: [0x78,0x72,0xa1,0x11]
# CHECK:        subs_u.b        $w21, $w6, $w25                 # encoding: [0x78,0x99,0x35,0x51]
# CHECK:        subs_u.h        $w3, $w10, $w7                  # encoding: [0x78,0xa7,0x50,0xd1]
# CHECK:        subs_u.w        $w9, $w15, $w10                 # encoding: [0x78,0xca,0x7a,0x51]
# CHECK:        subs_u.d        $w7, $w19, $w10                 # encoding: [0x78,0xea,0x99,0xd1]
# CHECK:        subsus_u.b      $w6, $w7, $w12                  # encoding: [0x79,0x0c,0x39,0x91]
# CHECK:        subsus_u.h      $w6, $w29, $w19                 # encoding: [0x79,0x33,0xe9,0x91]
# CHECK:        subsus_u.w      $w7, $w15, $w7                  # encoding: [0x79,0x47,0x79,0xd1]
# CHECK:        subsus_u.d      $w9, $w3, $w15                  # encoding: [0x79,0x6f,0x1a,0x51]
# CHECK:        subsuu_s.b      $w22, $w3, $w31                 # encoding: [0x79,0x9f,0x1d,0x91]
# CHECK:        subsuu_s.h      $w19, $w23, $w22                # encoding: [0x79,0xb6,0xbc,0xd1]
# CHECK:        subsuu_s.w      $w9, $w10, $w13                 # encoding: [0x79,0xcd,0x52,0x51]
# CHECK:        subsuu_s.d      $w5, $w6, $w0                   # encoding: [0x79,0xe0,0x31,0x51]
# CHECK:        subv.b          $w6, $w13, $w19                 # encoding: [0x78,0x93,0x69,0x8e]
# CHECK:        subv.h          $w4, $w25, $w12                 # encoding: [0x78,0xac,0xc9,0x0e]
# CHECK:        subv.w          $w27, $w27, $w11                # encoding: [0x78,0xcb,0xde,0xce]
# CHECK:        subv.d          $w9, $w24, $w10                 # encoding: [0x78,0xea,0xc2,0x4e]
# CHECK:        vshf.b          $w3, $w16, $w5                  # encoding: [0x78,0x05,0x80,0xd5]
# CHECK:        vshf.h          $w20, $w19, $w8                 # encoding: [0x78,0x28,0x9d,0x15]
# CHECK:        vshf.w          $w16, $w30, $w25                # encoding: [0x78,0x59,0xf4,0x15]
# CHECK:        vshf.d          $w19, $w11, $w15                # encoding: [0x78,0x6f,0x5c,0xd5]

                add_a.b         $w26, $w9, $w4
                add_a.h         $w23, $w27, $w31
                add_a.w         $w11, $w6, $w22
                add_a.d         $w6, $w10, $w0
                adds_a.b        $w19, $w24, $w19
                adds_a.h        $w25, $w6, $w4
                adds_a.w        $w25, $w17, $w27
                adds_a.d        $w15, $w18, $w26
                adds_s.b        $w29, $w11, $w19
                adds_s.h        $w5, $w23, $w26
                adds_s.w        $w16, $w14, $w13
                adds_s.d        $w2, $w14, $w28
                adds_u.b        $w3, $w17, $w14
                adds_u.h        $w10, $w30, $w4
                adds_u.w        $w15, $w18, $w20
                adds_u.d        $w30, $w10, $w9
                addv.b          $w24, $w20, $w21
                addv.h          $w4, $w13, $w27
                addv.w          $w19, $w11, $w14
                addv.d          $w2, $w21, $w31
                asub_s.b        $w23, $w16, $w3
                asub_s.h        $w22, $w17, $w25
                asub_s.w        $w24, $w1, $w9
                asub_s.d        $w13, $w12, $w12
                asub_u.b        $w10, $w29, $w11
                asub_u.h        $w18, $w9, $w15
                asub_u.w        $w10, $w19, $w31
                asub_u.d        $w17, $w10, $w0
                ave_s.b         $w2, $w5, $w1
                ave_s.h         $w16, $w19, $w9
                ave_s.w         $w17, $w31, $w5
                ave_s.d         $w27, $w25, $w10
                ave_u.b         $w16, $w19, $w9
                ave_u.h         $w28, $w28, $w11
                ave_u.w         $w11, $w12, $w11
                ave_u.d         $w30, $w19, $w28
                aver_s.b        $w26, $w16, $w2
                aver_s.h        $w31, $w27, $w27
                aver_s.w        $w28, $w18, $w25
                aver_s.d        $w29, $w21, $w27
                aver_u.b        $w29, $w26, $w3
                aver_u.h        $w18, $w18, $w9
                aver_u.w        $w17, $w25, $w29
                aver_u.d        $w22, $w22, $w19
                bclr.b          $w2, $w15, $w29
                bclr.h          $w16, $w21, $w28
                bclr.w          $w19, $w2, $w9
                bclr.d          $w27, $w31, $w4
                binsl.b         $w5, $w16, $w24
                binsl.h         $w30, $w5, $w10
                binsl.w         $w14, $w15, $w13
                binsl.d         $w23, $w20, $w12
                binsr.b         $w22, $w11, $w2
                binsr.h         $w0, $w26, $w6
                binsr.w         $w26, $w3, $w28
                binsr.d         $w0, $w0, $w21
                bneg.b          $w0, $w11, $w24
                bneg.h          $w28, $w16, $w4
                bneg.w          $w3, $w26, $w19
                bneg.d          $w13, $w29, $w15
                bset.b          $w31, $w5, $w31
                bset.h          $w14, $w12, $w6
                bset.w          $w31, $w9, $w12
                bset.d          $w5, $w22, $w5
                ceq.b           $w31, $w31, $w18
                ceq.h           $w10, $w27, $w9
                ceq.w           $w9, $w5, $w14
                ceq.d           $w5, $w17, $w0
                cle_s.b         $w23, $w4, $w9
                cle_s.h         $w22, $w27, $w19
                cle_s.w         $w30, $w26, $w10
                cle_s.d         $w18, $w5, $w10
                cle_u.b         $w1, $w25, $w0
                cle_u.h         $w7, $w0, $w29
                cle_u.w         $w25, $w18, $w1
                cle_u.d         $w6, $w0, $w30
                clt_s.b         $w25, $w2, $w21
                clt_s.h         $w2, $w19, $w9
                clt_s.w         $w23, $w8, $w16
                clt_s.d         $w7, $w30, $w12
                clt_u.b         $w2, $w31, $w13
                clt_u.h         $w16, $w31, $w23
                clt_u.w         $w3, $w24, $w9
                clt_u.d         $w7, $w0, $w1
                div_s.b         $w29, $w3, $w18
                div_s.h         $w17, $w16, $w13
                div_s.w         $w4, $w25, $w30
                div_s.d         $w31, $w9, $w20
                div_u.b         $w6, $w29, $w10
                div_u.h         $w24, $w21, $w14
                div_u.w         $w29, $w14, $w25
                div_u.d         $w31, $w1, $w21
                dotp_s.h        $w23, $w22, $w25
                dotp_s.w        $w20, $w14, $w5
                dotp_s.d        $w17, $w2, $w22
                dotp_u.h        $w13, $w2, $w6
                dotp_u.w        $w15, $w22, $w21
                dotp_u.d        $w4, $w16, $w26
                dpadd_s.h       $w1, $w28, $w22
                dpadd_s.w       $w10, $w1, $w12
                dpadd_s.d       $w3, $w21, $w27
                dpadd_u.h       $w17, $w5, $w20
                dpadd_u.w       $w24, $w8, $w16
                dpadd_u.d       $w15, $w29, $w16
                dpsub_s.h       $w4, $w11, $w12
                dpsub_s.w       $w4, $w7, $w6
                dpsub_s.d       $w31, $w12, $w28
                dpsub_u.h       $w4, $w25, $w17
                dpsub_u.w       $w19, $w25, $w16
                dpsub_u.d       $w7, $w10, $w26
                hadd_s.h        $w28, $w24, $w2
                hadd_s.w        $w24, $w17, $w11
                hadd_s.d        $w17, $w15, $w20
                hadd_u.h        $w12, $w29, $w17
                hadd_u.w        $w9, $w5, $w6
                hadd_u.d        $w1, $w20, $w6
                hsub_s.h        $w16, $w14, $w29
                hsub_s.w        $w9, $w13, $w11
                hsub_s.d        $w30, $w18, $w14
                hsub_u.h        $w7, $w12, $w14
                hsub_u.w        $w21, $w5, $w5
                hsub_u.d        $w11, $w12, $w31
                ilvev.b         $w18, $w16, $w30
                ilvev.h         $w14, $w0, $w13
                ilvev.w         $w12, $w25, $w22
                ilvev.d         $w30, $w27, $w3
                ilvl.b          $w29, $w3, $w21
                ilvl.h          $w27, $w10, $w17
                ilvl.w          $w6, $w1, $w0
                ilvl.d          $w3, $w16, $w24
                ilvod.b         $w11, $w5, $w20
                ilvod.h         $w18, $w13, $w31
                ilvod.w         $w29, $w16, $w24
                ilvod.d         $w22, $w12, $w29
                ilvr.b          $w4, $w30, $w6
                ilvr.h          $w28, $w19, $w29
                ilvr.w          $w18, $w20, $w21
                ilvr.d          $w23, $w30, $w12
                maddv.b         $w17, $w31, $w29
                maddv.h         $w7, $w24, $w9
                maddv.w         $w22, $w22, $w20
                maddv.d         $w30, $w26, $w20
                max_a.b         $w23, $w11, $w23
                max_a.h         $w20, $w5, $w30
                max_a.w         $w7, $w18, $w30
                max_a.d         $w8, $w8, $w31
                max_s.b         $w10, $w1, $w19
                max_s.h         $w15, $w29, $w17
                max_s.w         $w15, $w29, $w14
                max_s.d         $w25, $w24, $w3
                max_u.b         $w12, $w24, $w5
                max_u.h         $w5, $w6, $w7
                max_u.w         $w16, $w4, $w7
                max_u.d         $w26, $w12, $w24
                min_a.b         $w4, $w26, $w1
                min_a.h         $w12, $w13, $w31
                min_a.w         $w28, $w20, $w0
                min_a.d         $w12, $w20, $w19
                min_s.b         $w19, $w3, $w14
                min_s.h         $w27, $w21, $w8
                min_s.w         $w0, $w14, $w30
                min_s.d         $w6, $w8, $w21
                min_u.b         $w22, $w26, $w8
                min_u.h         $w7, $w27, $w12
                min_u.w         $w8, $w20, $w14
                min_u.d         $w26, $w14, $w15
                mod_s.b         $w18, $w1, $w26
                mod_s.h         $w31, $w30, $w28
                mod_s.w         $w2, $w6, $w13
                mod_s.d         $w21, $w27, $w22
                mod_u.b         $w16, $w7, $w13
                mod_u.h         $w24, $w8, $w7
                mod_u.w         $w30, $w2, $w17
                mod_u.d         $w31, $w2, $w25
                msubv.b         $w14, $w5, $w12
                msubv.h         $w6, $w7, $w30
                msubv.w         $w13, $w2, $w21
                msubv.d         $w16, $w14, $w27
                mulv.b          $w20, $w3, $w13
                mulv.h          $w27, $w26, $w14
                mulv.w          $w10, $w29, $w3
                mulv.d          $w7, $w19, $w29
                pckev.b         $w5, $w27, $w7
                pckev.h         $w1, $w4, $w27
                pckev.w         $w30, $w20, $w0
                pckev.d         $w6, $w1, $w15
                pckod.b         $w18, $w28, $w30
                pckod.h         $w26, $w5, $w8
                pckod.w         $w9, $w4, $w2
                pckod.d         $w30, $w22, $w20
                sld.b           $w5, $w23[$12]
                sld.h           $w1, $w23[$3]
                sld.w           $w20, $w8[$9]
                sld.d           $w7, $w23[$30]
                sll.b           $w3, $w0, $w17
                sll.h           $w17, $w27, $w3
                sll.w           $w16, $w7, $w6
                sll.d           $w9, $w0, $w26
                splat.b         $w28, $w1[$1]
                splat.h         $w2, $w11[$11]
                splat.w         $w22, $w0[$11]
                splat.d         $w0, $w0[$2]
                sra.b           $w28, $w4, $w17
                sra.h           $w13, $w9, $w3
                sra.w           $w27, $w21, $w19
                sra.d           $w30, $w8, $w23
                srar.b          $w19, $w18, $w18
                srar.h          $w7, $w23, $w8
                srar.w          $w1, $w12, $w2
                srar.d          $w21, $w7, $w14
                srl.b           $w12, $w3, $w19
                srl.h           $w23, $w31, $w20
                srl.w           $w18, $w27, $w11
                srl.d           $w3, $w12, $w26
                srlr.b          $w15, $w21, $w11
                srlr.h          $w21, $w13, $w19
                srlr.w          $w6, $w30, $w3
                srlr.d          $w1, $w2, $w14
                subs_s.b        $w25, $w15, $w1
                subs_s.h        $w28, $w25, $w22
                subs_s.w        $w10, $w12, $w21
                subs_s.d        $w4, $w20, $w18
                subs_u.b        $w21, $w6, $w25
                subs_u.h        $w3, $w10, $w7
                subs_u.w        $w9, $w15, $w10
                subs_u.d        $w7, $w19, $w10
                subsus_u.b      $w6, $w7, $w12
                subsus_u.h      $w6, $w29, $w19
                subsus_u.w      $w7, $w15, $w7
                subsus_u.d      $w9, $w3, $w15
                subsuu_s.b      $w22, $w3, $w31
                subsuu_s.h      $w19, $w23, $w22
                subsuu_s.w      $w9, $w10, $w13
                subsuu_s.d      $w5, $w6, $w0
                subv.b          $w6, $w13, $w19
                subv.h          $w4, $w25, $w12
                subv.w          $w27, $w27, $w11
                subv.d          $w9, $w24, $w10
                vshf.b          $w3, $w16, $w5
                vshf.h          $w20, $w19, $w8
                vshf.w          $w16, $w30, $w25
                vshf.d          $w19, $w11, $w15
