# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 -mattr=+msa \
# RUN:     -show-encoding 2>%t1
# RUN: FileCheck %s < %t1

    .set noat
    addvi.b $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    addvi.b $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    addvi.h $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    addvi.h $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    addvi.w $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    addvi.w $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    addvi.d $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    addvi.d $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    andi.b $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    andi.b $w1, $w2, 256     # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    bclri.b $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
    bclri.b $w1, $w2, 8      # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
    bclri.h $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 4-bit unsigned immediate
    bclri.h $w1, $w2, 16     # CHECK: :[[@LINE]]:23: error: expected 4-bit unsigned immediate
    bclri.w $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    bclri.w $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    bclri.d $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 6-bit unsigned immediate
    bclri.d $w1, $w2, 64     # CHECK: :[[@LINE]]:23: error: expected 6-bit unsigned immediate
    binsli.b $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    binsli.b $w1, $w2, 8     # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    binsli.h $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    binsli.h $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    binsli.w $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    binsli.w $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    binsli.d $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    binsli.d $w1, $w2, 64    # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    binsri.b $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    binsri.b $w1, $w2, 8     # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    binsri.h $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    binsri.h $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    binsri.w $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    binsri.w $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    binsri.d $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    binsri.d $w1, $w2, 64    # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    bmnzi.b $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 8-bit unsigned immediate
    bmnzi.b $w1, $w2, 256    # CHECK: :[[@LINE]]:23: error: expected 8-bit unsigned immediate
    bmzi.b $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    bmzi.b $w1, $w2, 256     # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    bnegi.b $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
    bnegi.b $w1, $w2, 8      # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
    bnegi.h $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 4-bit unsigned immediate
    bnegi.h $w1, $w2, 16     # CHECK: :[[@LINE]]:23: error: expected 4-bit unsigned immediate
    bnegi.w $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    bnegi.w $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    bnegi.d $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 6-bit unsigned immediate
    bnegi.d $w1, $w2, 64     # CHECK: :[[@LINE]]:23: error: expected 6-bit unsigned immediate
    bseli.b $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 8-bit unsigned immediate
    bseli.b $w1, $w2, 256    # CHECK: :[[@LINE]]:23: error: expected 8-bit unsigned immediate
    bseti.b $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
    bseti.b $w1, $w2, 8      # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
    bseti.h $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 4-bit unsigned immediate
    bseti.h $w1, $w2, 16     # CHECK: :[[@LINE]]:23: error: expected 4-bit unsigned immediate
    bseti.w $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    bseti.w $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    bseti.d $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 6-bit unsigned immediate
    bseti.d $w1, $w2, 64     # CHECK: :[[@LINE]]:23: error: expected 6-bit unsigned immediate
    ceqi.b $w1, $w2, -17     # CHECK: :[[@LINE]]:22: error: expected 5-bit signed immediate
    ceqi.b $w1, $w2, 16      # CHECK: :[[@LINE]]:22: error: expected 5-bit signed immediate
    ceqi.h $w1, $w2, -17     # CHECK: :[[@LINE]]:22: error: expected 5-bit signed immediate
    ceqi.h $w1, $w2, 16      # CHECK: :[[@LINE]]:22: error: expected 5-bit signed immediate
    ceqi.w $w1, $w2, -17     # CHECK: :[[@LINE]]:22: error: expected 5-bit signed immediate
    ceqi.w $w1, $w2, 16      # CHECK: :[[@LINE]]:22: error: expected 5-bit signed immediate
    ceqi.d $w1, $w2, -17     # CHECK: :[[@LINE]]:22: error: expected 5-bit signed immediate
    ceqi.d $w1, $w2, 16      # CHECK: :[[@LINE]]:22: error: expected 5-bit signed immediate
    clei_s.b $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clei_s.b $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clei_s.h $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clei_s.h $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clei_s.w $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clei_s.w $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clei_s.d $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clei_s.d $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clei_u.b $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clei_u.b $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clei_u.h $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clei_u.h $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clei_u.w $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clei_u.w $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clei_u.d $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clei_u.d $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clti_s.b $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clti_s.b $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clti_s.h $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clti_s.h $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clti_s.w $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clti_s.w $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clti_s.d $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clti_s.d $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    clti_u.b $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clti_u.b $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clti_u.h $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clti_u.h $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clti_u.w $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clti_u.w $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clti_u.d $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    clti_u.d $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    copy_s.b $2, $w9[-1]     # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    copy_s.b $2, $w9[16]     # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    copy_s.h $2, $w9[-1]     # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    copy_s.h $2, $w9[8]      # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    copy_s.w $2, $w9[-1]     # CHECK: :[[@LINE]]:22: error: expected 2-bit unsigned immediate
    copy_s.w $2, $w9[4]      # CHECK: :[[@LINE]]:22: error: expected 2-bit unsigned immediate
    copy_s.d $2, $w9[-1]     # CHECK: :[[@LINE]]:22: error: expected 1-bit unsigned immediate
    copy_s.d $2, $w9[2]      # CHECK: :[[@LINE]]:22: error: expected 1-bit unsigned immediate
    copy_u.b $2, $w9[-1]     # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    copy_u.b $2, $w9[16]     # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    copy_u.h $2, $w9[-1]     # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    copy_u.h $2, $w9[8]      # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    copy_u.w $2, $w9[-1]     # CHECK: :[[@LINE]]:22: error: expected 2-bit unsigned immediate
    copy_u.w $2, $w9[4]      # CHECK: :[[@LINE]]:22: error: expected 2-bit unsigned immediate
    insert.b $w9[-1], $2     # CHECK: :[[@LINE]]:18: error: expected 4-bit unsigned immediate
    insert.b $w9[16], $2     # CHECK: :[[@LINE]]:18: error: expected 4-bit unsigned immediate
    insert.h $w9[-1], $2     # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
    insert.h $w9[8], $2      # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
    insert.w $w9[-1], $2     # CHECK: :[[@LINE]]:18: error: expected 2-bit unsigned immediate
    insert.w $w9[4], $2      # CHECK: :[[@LINE]]:18: error: expected 2-bit unsigned immediate
    insert.d $w9[-1], $2     # CHECK: :[[@LINE]]:18: error: expected 1-bit unsigned immediate
    insert.d $w9[2], $2      # CHECK: :[[@LINE]]:18: error: expected 1-bit unsigned immediate
    insve.b $w25[-1], $w9[0] # CHECK: :[[@LINE]]:18: error: expected 4-bit unsigned immediate
    insve.b $w25[16], $w9[0] # CHECK: :[[@LINE]]:18: error: expected 4-bit unsigned immediate
    insve.h $w24[-1], $w2[0] # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
    insve.h $w24[8], $w2[0]  # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
    insve.w $w0[-1], $w13[0] # CHECK: :[[@LINE]]:17: error: expected 2-bit unsigned immediate
    insve.w $w0[4], $w13[0]  # CHECK: :[[@LINE]]:17: error: expected 2-bit unsigned immediate
    insve.d $w3[-1], $w18[0] # CHECK: :[[@LINE]]:17: error: expected 1-bit unsigned immediate
    insve.d $w3[2], $w18[0]  # CHECK: :[[@LINE]]:17: error: expected 1-bit unsigned immediate
    insve.b $w25[3], $w9[1]  # CHECK: :[[@LINE]]:26: error: expected '0'
    insve.h $w24[2], $w2[1]  # CHECK: :[[@LINE]]:26: error: expected '0'
    insve.w $w0[2], $w13[1]  # CHECK: :[[@LINE]]:26: error: expected '0'
    insve.d $w3[0], $w18[1]  # CHECK: :[[@LINE]]:26: error: expected '0'
    ld.b $w0, -513($2)       # CHECK: :[[@LINE]]:15: error: expected memory with 10-bit signed offset
    ld.b $w0, 512($2)        # CHECK: :[[@LINE]]:15: error: expected memory with 10-bit signed offset
    ld.h $w0, -1025($2)      # CHECK: :[[@LINE]]:15: error: expected memory with 11-bit signed offset and multiple of 2
    ld.h $w0, 1024($2)       # CHECK: :[[@LINE]]:15: error: expected memory with 11-bit signed offset and multiple of 2
    ld.w $w0, -2049($2)      # CHECK: :[[@LINE]]:15: error: expected memory with 12-bit signed offset and multiple of 4
    ld.w $w0, 2048($2)       # CHECK: :[[@LINE]]:15: error: expected memory with 12-bit signed offset and multiple of 4
    ld.d $w0, -4097($2)      # CHECK: :[[@LINE]]:15: error: expected memory with 13-bit signed offset and multiple of 8
    ld.d $w0, 4096($2)       # CHECK: :[[@LINE]]:15: error: expected memory with 13-bit signed offset and multiple of 8
    ldi.b $w1, -1025         # CHECK: :[[@LINE]]:16: error: expected 10-bit signed immediate
    ldi.b $w1, 1024          # CHECK: :[[@LINE]]:16: error: expected 10-bit signed immediate
    ldi.h $w1, -1025         # CHECK: :[[@LINE]]:16: error: expected 10-bit signed immediate
    ldi.h $w1, 1024          # CHECK: :[[@LINE]]:16: error: expected 10-bit signed immediate
    ldi.w $w1, -1025         # CHECK: :[[@LINE]]:16: error: expected 10-bit signed immediate
    ldi.w $w1, 1024          # CHECK: :[[@LINE]]:16: error: expected 10-bit signed immediate
    ldi.d $w1, -1025         # CHECK: :[[@LINE]]:16: error: expected 10-bit signed immediate
    ldi.d $w1, 1024          # CHECK: :[[@LINE]]:16: error: expected 10-bit signed immediate
    lsa $2, $3, $4, 0        # CHECK: :[[@LINE]]:21: error: expected immediate in range 1 .. 4
    lsa $2, $3, $4, 5        # CHECK: :[[@LINE]]:21: error: expected immediate in range 1 .. 4
    maxi_s.b $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    maxi_s.b $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    maxi_s.h $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    maxi_s.h $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    maxi_s.w $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    maxi_s.w $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    maxi_s.d $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    maxi_s.d $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    maxi_u.b $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    maxi_u.b $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    maxi_u.h $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    maxi_u.h $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    maxi_u.w $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    maxi_u.w $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    maxi_u.d $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    maxi_u.d $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    mini_s.b $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    mini_s.b $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    mini_s.h $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    mini_s.h $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    mini_s.w $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    mini_s.w $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    mini_s.d $w1, $w2, -17   # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    mini_s.d $w1, $w2, 16    # CHECK: :[[@LINE]]:24: error: expected 5-bit signed immediate
    mini_u.b $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    mini_u.b $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    mini_u.h $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    mini_u.h $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    mini_u.w $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    mini_u.w $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    mini_u.d $w1, $w2, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    mini_u.d $w1, $w2, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    nori.b $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    nori.b $w1, $w2, 256     # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    ori.b $w1, $w2, -1       # CHECK: :[[@LINE]]:21: error: expected 8-bit unsigned immediate
    ori.b $w1, $w2, 256      # CHECK: :[[@LINE]]:21: error: expected 8-bit unsigned immediate
    sat_s.b $w31, $w31, -1   # CHECK: :[[@LINE]]:25: error: expected 3-bit unsigned immediate
    sat_s.b $w31, $w31, 8    # CHECK: :[[@LINE]]:25: error: expected 3-bit unsigned immediate
    sat_s.h $w31, $w31, -1   # CHECK: :[[@LINE]]:25: error: expected 4-bit unsigned immediate
    sat_s.h $w31, $w31, 16   # CHECK: :[[@LINE]]:25: error: expected 4-bit unsigned immediate
    sat_s.w $w31, $w31, -1   # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
    sat_s.w $w31, $w31, 32   # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
    sat_s.d $w31, $w31, -1   # CHECK: :[[@LINE]]:25: error: expected 6-bit unsigned immediate
    sat_s.d $w31, $w31, 64   # CHECK: :[[@LINE]]:25: error: expected 6-bit unsigned immediate
    sat_u.b $w31, $w31, -1   # CHECK: :[[@LINE]]:25: error: expected 3-bit unsigned immediate
    sat_u.b $w31, $w31, 8    # CHECK: :[[@LINE]]:25: error: expected 3-bit unsigned immediate
    sat_u.h $w31, $w31, -1   # CHECK: :[[@LINE]]:25: error: expected 4-bit unsigned immediate
    sat_u.h $w31, $w31, 16   # CHECK: :[[@LINE]]:25: error: expected 4-bit unsigned immediate
    sat_u.w $w31, $w31, -1   # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
    sat_u.w $w31, $w31, 32   # CHECK: :[[@LINE]]:25: error: expected 5-bit unsigned immediate
    sat_u.d $w31, $w31, -1   # CHECK: :[[@LINE]]:25: error: expected 6-bit unsigned immediate
    sat_u.d $w31, $w31, 64   # CHECK: :[[@LINE]]:25: error: expected 6-bit unsigned immediate
    shf.b $w19, $w30, -1     # CHECK: :[[@LINE]]:23: error: expected 8-bit unsigned immediate
    shf.b $w19, $w30, 256    # CHECK: :[[@LINE]]:23: error: expected 8-bit unsigned immediate
    shf.h $w17, $w8, -1      # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    shf.h $w17, $w8, 256     # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    shf.w $w14, $w3, -1      # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    shf.w $w14, $w3, 256     # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    sldi.b $w0, $w29[-1]     # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    sldi.b $w0, $w29[16]     # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    sldi.h $w8, $w17[-1]     # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    sldi.h $w8, $w17[8]      # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    sldi.w $w20, $w27[-1]    # CHECK: :[[@LINE]]:23: error: expected 2-bit unsigned immediate
    sldi.w $w20, $w27[4]     # CHECK: :[[@LINE]]:23: error: expected 2-bit unsigned immediate
    sldi.d $w4, $w12[-1]     # CHECK: :[[@LINE]]:22: error: expected 1-bit unsigned immediate
    sldi.d $w4, $w12[2]      # CHECK: :[[@LINE]]:22: error: expected 1-bit unsigned immediate
    slli.b $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    slli.b $w1, $w2, 8       # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    slli.h $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    slli.h $w1, $w2, 16      # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    slli.w $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
    slli.w $w1, $w2, 32      # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
    slli.d $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
    slli.d $w1, $w2, 64      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
    splati.b $w0, $w29[-1]   # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    splati.b $w0, $w29[16]   # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    splati.h $w8, $w17[-1]   # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    splati.h $w8, $w17[8]    # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    splati.w $w2, $w27[-1]   # CHECK: :[[@LINE]]:24: error: expected 2-bit unsigned immediate
    splati.w $w2, $w27[4]    # CHECK: :[[@LINE]]:24: error: expected 2-bit unsigned immediate
    splati.d $w4, $w12[-1]   # CHECK: :[[@LINE]]:24: error: expected 1-bit unsigned immediate
    splati.d $w4, $w12[2]    # CHECK: :[[@LINE]]:24: error: expected 1-bit unsigned immediate
    srai.b $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    srai.b $w1, $w2, 8       # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    srai.h $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    srai.h $w1, $w2, 16      # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    srai.w $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
    srai.w $w1, $w2, 32      # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
    srai.d $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
    srai.d $w1, $w2, 64      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
    srari.b $w5, $w25, -1    # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    srari.b $w5, $w25, 8     # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    srari.h $w5, $w25, -1    # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    srari.h $w5, $w25, 16    # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    srari.w $w5, $w25, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    srari.w $w5, $w25, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    srari.d $w5, $w25, -1    # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    srari.d $w5, $w25, 64    # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    srli.b $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    srli.b $w1, $w2, 8       # CHECK: :[[@LINE]]:22: error: expected 3-bit unsigned immediate
    srli.h $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    srli.h $w1, $w2, 16      # CHECK: :[[@LINE]]:22: error: expected 4-bit unsigned immediate
    srli.w $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
    srli.w $w1, $w2, 32      # CHECK: :[[@LINE]]:22: error: expected 5-bit unsigned immediate
    srli.d $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
    srli.d $w1, $w2, 64      # CHECK: :[[@LINE]]:22: error: expected 6-bit unsigned immediate
    srlri.b $w18, $w3, -1    # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    srlri.b $w18, $w3, 8     # CHECK: :[[@LINE]]:24: error: expected 3-bit unsigned immediate
    srlri.h $w18, $w3, -1    # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    srlri.h $w18, $w3, 16    # CHECK: :[[@LINE]]:24: error: expected 4-bit unsigned immediate
    srlri.w $w18, $w3, -1    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    srlri.w $w18, $w3, 32    # CHECK: :[[@LINE]]:24: error: expected 5-bit unsigned immediate
    srlri.d $w18, $w3, -1    # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    srlri.d $w18, $w3, 64    # CHECK: :[[@LINE]]:24: error: expected 6-bit unsigned immediate
    st.b $w0, -513($2)       # CHECK: :[[@LINE]]:15: error: expected memory with 10-bit signed offset
    st.b $w0, 512($2)        # CHECK: :[[@LINE]]:15: error: expected memory with 10-bit signed offset
    st.h $w0, -1025($2)      # CHECK: :[[@LINE]]:15: error: expected memory with 11-bit signed offset and multiple of 2
    st.h $w0, 1024($2)       # CHECK: :[[@LINE]]:15: error: expected memory with 11-bit signed offset and multiple of 2
    st.w $w0, -2049($2)      # CHECK: :[[@LINE]]:15: error: expected memory with 12-bit signed offset and multiple of 4
    st.w $w0, 2048($2)       # CHECK: :[[@LINE]]:15: error: expected memory with 12-bit signed offset and multiple of 4
    st.d $w0, -4097($2)      # CHECK: :[[@LINE]]:15: error: expected memory with 13-bit signed offset and multiple of 8
    st.d $w0, 4096($2)       # CHECK: :[[@LINE]]:15: error: expected memory with 13-bit signed offset and multiple of 8
    subvi.b $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    subvi.b $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    subvi.h $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    subvi.h $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    subvi.w $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    subvi.w $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    subvi.d $w1, $w2, -1     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    subvi.d $w1, $w2, 32     # CHECK: :[[@LINE]]:23: error: expected 5-bit unsigned immediate
    xori.b $w1, $w2, -1      # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
    xori.b $w1, $w2, 256     # CHECK: :[[@LINE]]:22: error: expected 8-bit unsigned immediate
