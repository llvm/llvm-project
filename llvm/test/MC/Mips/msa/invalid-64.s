# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips64r2 -mattr=+msa \
# RUN:     -show-encoding 2>%t1
# RUN: FileCheck %s < %t1

    .set noat
    dlsa    $2, $3, $4, 0   # CHECK: :[[@LINE]]:25: error: expected immediate in range 1 .. 4
    dlsa    $2, $3, $4, 5   # CHECK: :[[@LINE]]:25: error: expected immediate in range 1 .. 4
    insve.b $w25[-1], $w9[0] # CHECK: :[[@LINE]]:18: error: expected 4-bit unsigned immediate
    insve.b $w25[16], $w9[0] # CHECK: :[[@LINE]]:18: error: expected 4-bit unsigned immediate
    insve.h $w24[-1], $w2[0] # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
    insve.h $w24[8], $w2[0]  # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
    insve.w $w0[-1], $w13[0] # CHECK: :[[@LINE]]:17: error: expected 2-bit unsigned immediate
    insve.w $w0[4], $w13[0]  # CHECK: :[[@LINE]]:17: error: expected 2-bit unsigned immediate
    insve.d $w3[-1], $w18[0] # CHECK: :[[@LINE]]:17: error: expected 1-bit unsigned immediate
    insve.d $w3[2], $w18[0]  # CHECK: :[[@LINE]]:17: error: expected 1-bit unsigned immediate
    insve.b $w25[3], $w9[1] # CHECK: :[[@LINE]]:26: error: expected '0'
    insve.h $w24[2], $w2[1] # CHECK: :[[@LINE]]:26: error: expected '0'
    insve.w $w0[2], $w13[1] # CHECK: :[[@LINE]]:26: error: expected '0'
    insve.d $w3[0], $w18[1] # CHECK: :[[@LINE]]:26: error: expected '0'
    lsa     $2, $3, $4, 0   # CHECK: :[[@LINE]]:25: error: expected immediate in range 1 .. 4
    lsa     $2, $3, $4, 5   # CHECK: :[[@LINE]]:25: error: expected immediate in range 1 .. 4
    sat_s.b $w31, $w31, -1  # CHECK: :[[@LINE]]:25: error: expected 3-bit unsigned immediate
    sat_s.b $w31, $w31, 8   # CHECK: :[[@LINE]]:25: error: expected 3-bit unsigned immediate
    sat_s.h $w31, $w31, -1  # CHECK: :[[@LINE]]:25: error: expected 4-bit unsigned immediate
    sat_s.h $w31, $w31, 16  # CHECK: :[[@LINE]]:25: error: expected 4-bit unsigned immediate
    sat_s.w $w31, $w31, -1  # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
    sat_s.w $w31, $w31, 32  # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
    sat_s.d $w31, $w31, -1  # CHECK: :[[@LINE]]:25: error: expected 6-bit unsigned immediate
    sat_s.d $w31, $w31, 64  # CHECK: :[[@LINE]]:25: error: expected 6-bit unsigned immediate
    sat_u.b $w31, $w31, -1  # CHECK: :[[@LINE]]:25: error: expected 3-bit unsigned immediate
    sat_u.b $w31, $w31, 8   # CHECK: :[[@LINE]]:25: error: expected 3-bit unsigned immediate
    sat_u.h $w31, $w31, -1  # CHECK: :[[@LINE]]:25: error: expected 4-bit unsigned immediate
    sat_u.h $w31, $w31, 16  # CHECK: :[[@LINE]]:25: error: expected 4-bit unsigned immediate
    sat_u.w $w31, $w31, -1  # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
    sat_u.w $w31, $w31, 32  # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
    sat_u.d $w31, $w31, -1  # CHECK: :[[@LINE]]:25: error: expected 6-bit unsigned immediate
    sat_u.d $w31, $w31, 64  # CHECK: :[[@LINE]]:25: error: expected 6-bit unsigned immediate
    shf.b $w19, $w30, -1    # CHECK: :[[@LINE]]:23: error: expected 8-bit unsigned immediate
    shf.h $w17, $w8, -1     # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    shf.w $w14, $w3, -1     # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    sldi.b $w0, $w29[-1]    # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    sldi.b $w0, $w29[16]    # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    sldi.d $w4, $w12[-1]    # CHECK: :[[@LINE]]:22: error: expected 1-bit unsigned immediate
    sldi.d $w4, $w12[2]     # CHECK: :[[@LINE]]:22: error: expected 1-bit unsigned immediate
    sldi.h $w8, $w17[-1]    # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    sldi.h $w8, $w17[8]     # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    sldi.w $w20, $w27[-1]   # CHECK: :[[@LINE]]:23: error: expected 2-bit unsigned immediate
    sldi.w $w20, $w27[4]    # CHECK: :[[@LINE]]:23: error: expected 2-bit unsigned immediate
    srari.b $w5, $w25, -1   # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    srari.b $w5, $w25, 8    # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    srari.h $w5, $w25, -1   # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    srari.h $w5, $w25, 16   # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    srari.w $w5, $w25, -1   # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    srari.w $w5, $w25, 32   # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    srari.d $w5, $w25, -1   # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    srari.d $w5, $w25, 64   # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    srlri.b $w18, $w3, -1   # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    srlri.b $w18, $w3, 8    # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    srlri.h $w18, $w3, -1   # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    srlri.h $w18, $w3, 16   # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    srlri.w $w18, $w3, -1   # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    srlri.w $w18, $w3, 32   # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    srlri.d $w18, $w3, -1   # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    srlri.d $w18, $w3, 64   # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
