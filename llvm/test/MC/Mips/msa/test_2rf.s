# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        fclass.w        $w26, $w12              # encoding: [0x7b,0x20,0x66,0x9e]
# CHECK:        fclass.d        $w24, $w17              # encoding: [0x7b,0x21,0x8e,0x1e]
# CHECK:        fexupl.w        $w8, $w0                # encoding: [0x7b,0x30,0x02,0x1e]
# CHECK:        fexupl.d        $w17, $w29              # encoding: [0x7b,0x31,0xec,0x5e]
# CHECK:        fexupr.w        $w13, $w4               # encoding: [0x7b,0x32,0x23,0x5e]
# CHECK:        fexupr.d        $w5, $w2                # encoding: [0x7b,0x33,0x11,0x5e]
# CHECK:        ffint_s.w       $w20, $w29              # encoding: [0x7b,0x3c,0xed,0x1e]
# CHECK:        ffint_s.d       $w12, $w15              # encoding: [0x7b,0x3d,0x7b,0x1e]
# CHECK:        ffint_u.w       $w7, $w27               # encoding: [0x7b,0x3e,0xd9,0xde]
# CHECK:        ffint_u.d       $w19, $w16              # encoding: [0x7b,0x3f,0x84,0xde]
# CHECK:        ffql.w          $w31, $w13              # encoding: [0x7b,0x34,0x6f,0xde]
# CHECK:        ffql.d          $w12, $w13              # encoding: [0x7b,0x35,0x6b,0x1e]
# CHECK:        ffqr.w          $w27, $w30              # encoding: [0x7b,0x36,0xf6,0xde]
# CHECK:        ffqr.d          $w30, $w15              # encoding: [0x7b,0x37,0x7f,0x9e]
# CHECK:        flog2.w         $w25, $w31              # encoding: [0x7b,0x2e,0xfe,0x5e]
# CHECK:        flog2.d         $w18, $w10              # encoding: [0x7b,0x2f,0x54,0x9e]
# CHECK:        frint.w         $w7, $w15               # encoding: [0x7b,0x2c,0x79,0xde]
# CHECK:        frint.d         $w21, $w22              # encoding: [0x7b,0x2d,0xb5,0x5e]
# CHECK:        frcp.w          $w19, $w0               # encoding: [0x7b,0x2a,0x04,0xde]
# CHECK:        frcp.d          $w4, $w14               # encoding: [0x7b,0x2b,0x71,0x1e]
# CHECK:        frsqrt.w        $w12, $w17              # encoding: [0x7b,0x28,0x8b,0x1e]
# CHECK:        frsqrt.d        $w23, $w11              # encoding: [0x7b,0x29,0x5d,0xde]
# CHECK:        fsqrt.w         $w0, $w11               # encoding: [0x7b,0x26,0x58,0x1e]
# CHECK:        fsqrt.d         $w15, $w12              # encoding: [0x7b,0x27,0x63,0xde]
# CHECK:        ftint_s.w       $w30, $w5               # encoding: [0x7b,0x38,0x2f,0x9e]
# CHECK:        ftint_s.d       $w5, $w23               # encoding: [0x7b,0x39,0xb9,0x5e]
# CHECK:        ftint_u.w       $w20, $w14              # encoding: [0x7b,0x3a,0x75,0x1e]
# CHECK:        ftint_u.d       $w23, $w21              # encoding: [0x7b,0x3b,0xad,0xde]
# CHECK:        ftrunc_s.w      $w29, $w17              # encoding: [0x7b,0x22,0x8f,0x5e]
# CHECK:        ftrunc_s.d      $w12, $w27              # encoding: [0x7b,0x23,0xdb,0x1e]
# CHECK:        ftrunc_u.w      $w17, $w15              # encoding: [0x7b,0x24,0x7c,0x5e]
# CHECK:        ftrunc_u.d      $w5, $w27               # encoding: [0x7b,0x25,0xd9,0x5e]

                fclass.w        $w26, $w12
                fclass.d        $w24, $w17
                fexupl.w        $w8, $w0
                fexupl.d        $w17, $w29
                fexupr.w        $w13, $w4
                fexupr.d        $w5, $w2
                ffint_s.w       $w20, $w29
                ffint_s.d       $w12, $w15
                ffint_u.w       $w7, $w27
                ffint_u.d       $w19, $w16
                ffql.w          $w31, $w13
                ffql.d          $w12, $w13
                ffqr.w          $w27, $w30
                ffqr.d          $w30, $w15
                flog2.w         $w25, $w31
                flog2.d         $w18, $w10
                frint.w         $w7, $w15
                frint.d         $w21, $w22
                frcp.w          $w19, $w0
                frcp.d          $w4, $w14
                frsqrt.w        $w12, $w17
                frsqrt.d        $w23, $w11
                fsqrt.w         $w0, $w11
                fsqrt.d         $w15, $w12
                ftint_s.w       $w30, $w5
                ftint_s.d       $w5, $w23
                ftint_u.w       $w20, $w14
                ftint_u.d       $w23, $w21
                ftrunc_s.w      $w29, $w17
                ftrunc_s.d      $w12, $w27
                ftrunc_u.w      $w17, $w15
                ftrunc_u.d      $w5, $w27
