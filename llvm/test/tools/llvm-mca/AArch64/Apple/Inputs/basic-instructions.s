#------------------------------------------------------------------------------
# Add/sub (immediate)
#------------------------------------------------------------------------------

add      w2, w3, #4095
add      w30, w29, #1, lsl #12
add      w13, w5, #4095, lsl #12
add      x5, x7, #1638
add      w20, wsp, #801
add      wsp, wsp, #1104
add      wsp, w30, #4084
add      x0, x24, #291
add      x3, x24, #4095, lsl #12
add      x8, sp, #1074
add      sp, x29, #3816
sub      w0, wsp, #4077
sub      w4, w20, #546, lsl #12
sub      sp, sp, #288
sub      wsp, w19, #16
adds     w13, w23, #291, lsl #12
cmn      w2, #4095
adds     w20, wsp, #0
cmn      x3, #1, lsl #12
cmp      wsp, #2342
cmp      sp, #20, lsl #12
cmp      x30, #4095
subs     x4, sp, #3822
cmn      w3, #291, lsl #12
cmn      wsp, #1365
cmn      sp, #1092, lsl #12
mov      x10, #-63432

#------------------------------------------------------------------------------
# Add-subtract (shifted register)
#------------------------------------------------------------------------------

add      wsp, wsp, w10
add      x25, x9, w25, uxtb
add      w3, w5, w7
add      wzr, w3, w5
add      w20, wzr, w4
add      w4, w6, wzr
add      w11, w13, w15
add      w9, w3, wzr, lsl #1
add      w17, w29, w20, lsl #31
add      w21, w22, w23, lsr #0
add      w24, w25, w26, lsr #18
add      w27, w28, w29, lsr #31
add      w2, w3, w4, asr #0
add      w5, w6, w7, asr #21
add      w8, w9, w10, asr #31
add      x3, x5, x7
add      xzr, x3, x5
add      x20, xzr, x4
add      x4, x6, xzr
add      x11, x13, x15
add      x9, x3, xzr, lsl #10
add      x17, x29, x20, lsl #3
add      x21, x22, x23, lsr #0
add      x24, x25, x26, lsr #18
add      x27, x28, x29, lsr #63
add      x2, x3, x4, asr #0
add      x5, x6, x7, asr #21
add      x8, x9, x10, asr #63
adds     w3, w5, w7
adds     w17, wsp, w25
adds     x13, x23, w8, uxtb
cmn      w3, w5
adds     w20, wzr, w4
adds     w4, w6, wzr
adds     w11, w13, w15
adds     w9, w3, wzr, lsl #1
adds     w17, w29, w20, lsl #31
adds     w21, w22, w23, lsr #0
adds     w24, w25, w26, lsr #18
adds     w27, w28, w29, lsr #31
adds     w2, w3, w4, asr #0
adds     w5, w6, w7, asr #21
adds     w8, w9, w10, asr #31
adds     x3, x5, x7
cmn      x3, x5
adds     x20, xzr, x4
adds     x4, x6, xzr
adds     x11, x13, x15
adds     x9, x3, xzr, lsl #10
adds     x17, x29, x20, lsl #3
adds     x21, x22, x23, lsr #0
adds     x24, x25, x26, lsr #18
adds     x27, x28, x29, lsr #63
adds     x2, x3, x4, asr #0
adds     x5, x6, x7, asr #21
adds     x8, x9, x10, asr #63
sub      w3, w5, w7
sub      wzr, w3, w5
sub      w4, w6, wzr
sub      w11, w13, w15
sub      w9, w3, wzr, lsl #1
sub      w17, w29, w20, lsl #31
sub      w21, w22, w23, lsr #0
sub      w24, w25, w26, lsr #18
sub      w27, w28, w29, lsr #31
sub      w2, w3, w4, asr #0
sub      w5, w6, w7, asr #21
sub      w8, w9, w10, asr #31
sub      x3, x5, x7
sub      xzr, x3, x5
sub      x4, x6, xzr
sub      x11, x13, x15
sub      x9, x3, xzr, lsl #10
sub      x17, x29, x20, lsl #3
sub      x21, x22, x23, lsr #0
sub      x24, x25, x26, lsr #18
sub      x27, x28, x29, lsr #63
sub      x2, x3, x4, asr #0
sub      x5, x6, x7, asr #21
sub      x8, x9, x10, asr #63
sub      w13, wsp, w10
sub      x16, x2, w19, uxtb
subs     x13, x15, x14, sxtx #1
subs     w3, w5, w7
cmp      w3, w5
subs     w4, w6, wzr
subs     w11, w13, w15
subs     w9, w3, wzr, lsl #1
subs     w17, w29, w20, lsl #31
subs     w21, w22, w23, lsr #0
subs     w24, w25, w26, lsr #18
subs     w27, w28, w29, lsr #31
subs     w2, w3, w4, asr #0
subs     w5, w6, w7, asr #21
subs     w8, w9, w10, asr #31
subs     x3, x5, x7
cmp      x3, x5
subs     x4, x6, xzr
subs     x11, x13, x15
subs     x9, x3, xzr, lsl #10
subs     x17, x29, x20, lsl #3
subs     x21, x22, x23, lsr #0
subs     x24, x25, x26, lsr #18
subs     x27, x28, x29, lsr #63
subs     x2, x3, x4, asr #0
subs     x5, x6, x7, asr #21
subs     x8, x9, x10, asr #63
cmn      wzr, w4
cmn      w5, wzr
cmn      w6, w7
cmn      w8, w9, lsl #1
cmn      w10, w11, lsl #31
cmn      w12, w13, lsr #0
cmn      w14, w15, lsr #21
cmn      w16, w17, lsr #31
cmn      w18, w19, asr #0
cmn      w20, w21, asr #22
cmn      w22, w23, asr #31
cmn      x0, x3
cmn      xzr, x4
cmn      x5, xzr
cmn      x6, x7
cmn      x8, x9, lsl #15
cmn      x10, x11, lsl #3
cmn      x12, x13, lsr #0
cmn      x14, x15, lsr #41
cmn      x16, x17, lsr #63
cmn      x18, x19, asr #0
cmn      x20, x21, asr #55
cmn      x22, x23, asr #63
cmp      w0, w3
cmp      wzr, w4
cmp      w5, wzr
cmp      w6, w7
cmp      w8, w9, lsl #1
cmp      w10, w11, lsl #31
cmp      w12, w13, lsr #0
cmp      w14, w15, lsr #21
cmp      w18, w19, asr #0
cmp      w20, w21, asr #22
cmp      w22, w23, asr #31
cmp      wsp, w26
cmp      x16, w27, uxtb
cmp      x0, x3
cmp      xzr, x4
cmp      x5, xzr
cmp      x6, x7
cmp      x8, x9, lsl #15
cmp      x10, x11, lsl #3
cmp      x12, x13, lsr #0
cmp      x14, x15, lsr #41
cmp      x16, x17, lsr #63
cmp      x18, x19, asr #0
cmp      x20, x21, asr #55
cmp      x22, x23, asr #63
cmp      wzr, w0
cmp      xzr, x0
mov      sp, x30
mov      wsp, w20
mov      x11, sp
mov      w24, wsp

#------------------------------------------------------------------------------
# Add-subtract (shifted register)
#------------------------------------------------------------------------------

adc      w29, w27, w25
adc      wzr, w3, w4
adc      w9, wzr, w10
adc      w20, w0, wzr
adc      x29, x27, x25
adc      xzr, x3, x4
adc      x9, xzr, x10
adc      x20, x0, xzr
adcs     w29, w27, w25
adcs     wzr, w3, w4
adcs     w9, wzr, w10
adcs     w20, w0, wzr
adcs     x29, x27, x25
adcs     xzr, x3, x4
adcs     x9, xzr, x10
adcs     x20, x0, xzr
sbc      w29, w27, w25
sbc      wzr, w3, w4
ngc      w9, w10
sbc      w20, w0, wzr
sbc      x29, x27, x25
sbc      xzr, x3, x4
ngc      x9, x10
sbc      x20, x0, xzr
sbcs     w29, w27, w25
sbcs     wzr, w3, w4
ngcs     w9, w10
sbcs     w20, w0, wzr
sbcs     x29, x27, x25
sbcs     xzr, x3, x4
ngcs     x9, x10
sbcs     x20, x0, xzr
ngc      w3, w12
ngc      wzr, w9
ngc      w23, wzr
ngc      x29, x30
ngc      xzr, x0
ngc      x0, xzr
ngcs     w3, w12
ngcs     wzr, w9
ngcs     w23, wzr
ngcs     x29, x30
ngcs     xzr, x0
ngcs     x0, xzr

#------------------------------------------------------------------------------
# Compare and branch (immediate)
#------------------------------------------------------------------------------

sbfx     x1, x2, #3, #2
asr      x3, x4, #63
asr      wzr, wzr, #31
sbfx     w12, w9, #0, #1
ubfiz    x4, x5, #52, #11
ubfx     xzr, x4, #0, #1
ubfiz    x4, xzr, #1, #6
lsr      x5, x6, #12
bfi      x4, x5, #52, #11
bfxil    xzr, x4, #0, #1
bfi      x4, xzr, #1, #6
bfxil    x5, x6, #12, #52
sxtb     w1, w2
sxtb     xzr, w3
sxth     w9, w10
sxth     x0, w1
sxtw     x3, w30
uxtb     w1, w2
uxth     w9, w10
ubfx     x3, x30, #0, #32
asr      w3, w2, #0
asr      w9, w10, #31
asr      x20, x21, #63
asr      w1, wzr, #3
lsr      w3, w2, #0
lsr      w9, w10, #31
lsr      x20, x21, #63
lsr      wzr, wzr, #3
lsl      w9, w10, #31
lsl      x20, x21, #63
lsl      w1, wzr, #3
sbfiz    x2, x3, #63, #1
sbfiz    x9, x10, #5, #59
sbfiz    w11, w12, #31, #1
sbfiz    w13, w14, #29, #3
sbfiz    xzr, xzr, #10, #11
sbfx     w9, w10, #0, #1
asr      x2, x3, #63
asr      x19, x20, #0
asr      x9, x10, #5
asr      w9, w10, #0
asr      w11, w12, #31
asr      w13, w14, #29
sbfx     xzr, xzr, #10, #11
bfi      x2, x3, #63, #1
bfi      x9, x10, #5, #59
bfi      w11, w12, #31, #1
bfi      w13, w14, #29, #3
bfi      xzr, xzr, #10, #11
bfxil    w9, w10, #0, #1
bfxil    x2, x3, #63, #1
bfxil    x19, x20, #0, #64
bfxil    x9, x10, #5, #59
bfxil    w9, w10, #0, #32
bfxil    w11, w12, #31, #1
bfxil    w13, w14, #29, #3
bfxil    xzr, xzr, #10, #11
lsl      x2, x3, #63
lsl      x9, x10, #5
lsl      w11, w12, #31
lsl      w13, w14, #29
ubfiz    xzr, xzr, #10, #11
ubfx     w9, w10, #0, #1
lsr      x2, x3, #63
lsr      x19, x20, #0
lsr      x9, x10, #5
lsr      w9, w10, #0
lsr      w11, w12, #31
lsr      w13, w14, #29
ubfx     xzr, xzr, #10, #11

#------------------------------------------------------------------------------
# Compare and branch (immediate)
#------------------------------------------------------------------------------

cbz      w5, #4
cbz      x5, #0
cbnz     x2, #-4
cbnz     x26, #1048572
cbz      wzr, #0
cbnz     xzr, #0
cbnz     w21, test

#------------------------------------------------------------------------------
# Conditional branch (immediate)
#------------------------------------------------------------------------------

b.ne #4
b.ge #1048572
b.ge #-4

#------------------------------------------------------------------------------
# Conditional compare (immediate)
#------------------------------------------------------------------------------

ccmp w1, #31, #0, eq
ccmp w3, #0, #15, hs
ccmp wzr, #15, #13, hs
ccmp x9, #31, #0, le
ccmp x3, #0, #15, gt
ccmp xzr, #5, #7, ne
ccmn w1, #31, #0, eq
ccmn w3, #0, #15, hs
ccmn wzr, #15, #13, hs
ccmn x9, #31, #0, le
ccmn x3, #0, #15, gt
ccmn xzr, #5, #7, ne

#------------------------------------------------------------------------------
# Conditional compare (register)
#------------------------------------------------------------------------------

ccmp w1, wzr, #0, eq
ccmp w3, w0, #15, hs
ccmp wzr, w15, #13, hs
ccmp x9, xzr, #0, le
ccmp x3, x0, #15, gt
ccmp xzr, x5, #7, ne
ccmn w1, wzr, #0, eq
ccmn w3, w0, #15, hs
ccmn wzr, w15, #13, hs
ccmn x9, xzr, #0, le
ccmn x3, x0, #15, gt
ccmn xzr, x5, #7, ne

#------------------------------------------------------------------------------
# Conditional branch (immediate)
#------------------------------------------------------------------------------

csel     w1, w0, w19, ne
csel     wzr, w5, w9, eq
csel     w9, wzr, w30, gt
csel     w1, w28, wzr, mi
csel     x19, x23, x29, lt
csel     xzr, x3, x4, ge
csel     x5, xzr, x6, hs
csel     x7, x8, xzr, lo
csinc    w1, w0, w19, ne
csinc    wzr, w5, w9, eq
csinc    w9, wzr, w30, gt
csinc    w1, w28, wzr, mi
csinc    x19, x23, x29, lt
csinc    xzr, x3, x4, ge
csinc    x5, xzr, x6, hs
csinc    x7, x8, xzr, lo
csinv    w1, w0, w19, ne
csinv    wzr, w5, w9, eq
csinv    w9, wzr, w30, gt
csinv    w1, w28, wzr, mi
csinv    x19, x23, x29, lt
csinv    xzr, x3, x4, ge
csinv    x5, xzr, x6, hs
csinv    x7, x8, xzr, lo
csneg    w1, w0, w19, ne
csneg    wzr, w5, w9, eq
csneg    w9, wzr, w30, gt
csneg    w1, w28, wzr, mi
csneg    x19, x23, x29, lt
csneg    xzr, x3, x4, ge
csneg    x5, xzr, x6, hs
csneg    x7, x8, xzr, lo
cset    w3, eq
cset    x9, pl
csetm    w20, ne
csetm    x30, ge
csinc    w2, wzr, wzr, al
csinv    x3, xzr, xzr, nv
cinc    w3, w5, gt
cinc    wzr, w4, le
cset    w9, lt
cinc    x3, x5, gt
cinc    xzr, x4, le
cset    x9, lt
csinc   w5, w6, w6, nv
csinc   x1, x2, x2, al
cinv    w3, w5, gt
cinv    wzr, w4, le
csetm   w9, lt
cinv    x3, x5, gt
cinv    xzr, x4, le
csetm   x9, lt
csinv   x1, x0, x0, al
csinv   w9, w8, w8, nv
cneg     w3, w5, gt
cneg     wzr, w4, le
cneg     w9, wzr, lt
cneg     x3, x5, gt
cneg     xzr, x4, le
cneg     x9, xzr, lt
csneg    x4, x8, x8, al

#------------------------------------------------------------------------------
# Data-processing (1 source)
#------------------------------------------------------------------------------

rbit	w0, w7
rbit   x18, x3
rev16	w17, w1
rev16	x5, x2
rev	w18, w0
rev32	x20, x1
rev	x22, x2
clz	w24, w3
clz	x26, x4
cls	w3, w5
cls	x20, x5

#------------------------------------------------------------------------------
# Data-processing (2 source)
#------------------------------------------------------------------------------

udiv	w0, w7, w10
udiv	x9, x22, x4
sdiv	w12, w21, w0
sdiv	x13, x2, x1
lsl	w11, w12, w13
lsl	x14, x15, x16
lsr	w17, w18, w19
lsr	x20, x21, x22
asr	w23, w24, w25
asr	x26, x27, x28
ror	w0, w1, w2
ror     x3, x4, x5
lsl	w6, w7, w8
lsl	x9, x10, x11
lsr	w12, w13, w14
lsr	x15, x16, x17
asr	w18, w19, w20
asr	x21, x22, x23
ror	w24, w25, w26
ror	x27, x28, x29

#------------------------------------------------------------------------------
# Data-processing (3 sources)
#------------------------------------------------------------------------------

crc32cb  w30, w23, w15
crc32cb  w31, w12, w14
crc32cb  w28, w10, w11
crc32b   w27, w12, w15
crc32h   w3, w15, w21
crc32w   w9, w18, w24
crc32x   w19, w6, x25
crc32ch  w25, w26, w16
crc32cw  w27, w12, w23
crc32cx  w21, w28, x5
smulh    x30, x29, x28
smulh    xzr, x27, x26
umulh    x30, x29, x28
umulh    x23, x30, xzr
madd     w1, w3, w7, w4
madd     wzr, w0, w9, w11
madd     w13, wzr, w4, w4
madd     w19, w30, wzr, w29
mul      w4, w5, w6
madd     x1, x3, x7, x4
madd     xzr, x0, x9, x11
madd     x13, xzr, x4, x4
madd     x19, x30, xzr, x29
mul      x4, x5, x6
msub     w1, w3, w7, w4
msub     wzr, w0, w9, w11
msub     w13, wzr, w4, w4
msub     w19, w30, wzr, w29
mneg     w4, w5, w6
msub     x1, x3, x7, x4
msub     xzr, x0, x9, x11
msub     x13, xzr, x4, x4
msub     x19, x30, xzr, x29
mneg     x4, x5, x6
smaddl   x3, w5, w2, x9
smaddl   xzr, w10, w11, x12
smaddl   x13, wzr, w14, x15
smaddl   x16, w17, wzr, x18
smull    x19, w20, w21
smsubl   x3, w5, w2, x9
smsubl   xzr, w10, w11, x12
smsubl   x13, wzr, w14, x15
smsubl   x16, w17, wzr, x18
smnegl   x19, w20, w21
umaddl   x3, w5, w2, x9
umaddl   xzr, w10, w11, x12
umaddl   x13, wzr, w14, x15
umaddl   x16, w17, wzr, x18
umull    x19, w20, w21
umsubl   x3, w5, w2, x9
umsubl   x16, w17, wzr, x18
umnegl   x19, w20, w21
smulh    x23, x22, xzr
umulh    x23, x22, xzr
mul      x19, x20, xzr
mneg     w21, w22, w23
smull    x11, w13, w17
umull    x11, w13, w17
smnegl   x11, w13, w17
umnegl   x11, w13, w17

#------------------------------------------------------------------------------
# Extract (immediate)
#------------------------------------------------------------------------------

extr     w3, w5, w7, #0
extr     w11, w13, w17, #31
extr     x3, x5, x7, #15
extr     x11, x13, x17, #63
ror      x19, x23, #24
ror      x29, xzr, #63
ror      w9, w13, #31

#------------------------------------------------------------------------------
# Floating-point compare
#------------------------------------------------------------------------------

fcmp    h5, h21
fcmp    h5, #0.0
fcmpe   h22, h21
fcmpe   h13, #0.0
fcmp    s3, s5
fcmp    s31, #0.0
fcmpe   s29, s30
fcmpe   s15, #0.0
fcmp    d4, d12
fcmp    d23, #0.0
fcmpe   d26, d22
fcmpe   d29, #0.0

#------------------------------------------------------------------------------
# Floating-point conditional compare
#------------------------------------------------------------------------------

fccmp s1, s31, #0, eq
fccmp s3, s0, #15, hs
fccmp s31, s15, #13, hs
fccmp d9, d31, #0, le
fccmp d3, d0, #15, gt
fccmp d31, d5, #7, ne
fccmp h31, h3, #11, hs
fccmpe h6, h1, #12, ne
fccmpe s1, s31, #0, eq
fccmpe s3, s0, #15, hs
fccmpe s31, s15, #13, hs
fccmpe d9, d31, #0, le
fccmpe d3, d0, #15, gt
fccmpe d31, d5, #7, ne

#-------------------------------------------------------------------------------
# Floating-point conditional compare
#-------------------------------------------------------------------------------

fcsel s3, s20, s9, pl
fcsel d9, d10, d11, mi
fcsel h26, h2, h11, hs

#------------------------------------------------------------------------------
# Floating-point data-processing (1 source)
#------------------------------------------------------------------------------

fmov     h18, h28
fmov     s0, s1
fabs     s2, s3
fneg     h2, h9
fneg     s4, s5
fsqrt    s6, s7
fcvt     d8, s9
fcvt     h10, s11
frintn   h12, h3
frintn   s12, s13
frintp   h17, h31
frintp   s14, s15
frintm   h0, h21
frintm   s16, s17
frintz   h10, h29
frintz   s18, s19
frinta   h22, h10
frinta   s20, s21
frintx   h4, h5
frintx   s22, s23
frinti   s24, s25
frinti   h31, h14
fmov     d0, d1
fabs     d2, d3
fneg     d4, d5
fsqrt    h13, h24
fsqrt    d6, d7
fcvt     s8, d9
fcvt     h10, d11
frintn   d12, d13
frintp   d14, d15
frintm   d16, d17
frintz   d18, d19
frinta   d20, d21
frintx   d22, d23
frinti   d24, d25
fcvt     s26, h27
fcvt     d28, h29

#------------------------------------------------------------------------------
# Floating-point data-processing (2 sources)
#------------------------------------------------------------------------------

fmul     s20, s19, s17
fdiv     h1, h26, h23
fdiv     s1, s2, s3
fadd     h23, h27, h22
fadd     s4, s5, s6
fsub     h20, h11, h18
fsub     s7, s8, s9
fmax     s10, s11, s12
fmax     h8, h7, h11
fmin     s13, s14, s15
fmaxnm   h29, h13, h14
fmaxnm   s16, s17, s18
fminnm   s19, s20, s21
fnmul    h3, h15, h7
fnmul    s22, s23, s2
fmul     d20, d19, d17
fdiv     d1, d2, d3
fadd     d4, d5, d6
fsub     d7, d8, d9
fmax     d10, d11, d12
fmin     d13, d14, d15
fmin     h4, h13, h17
fmaxnm   d16, d17, d18
fminnm   d19, d20, d21
fminnm   h29, h23, h17
fnmul    d22, d23, d24

#------------------------------------------------------------------------------
# Floating-point data-processing (1 source)
#------------------------------------------------------------------------------

fmadd h27, h0, h6, h28
fmadd s3, s5, s6, s31
fmadd d3, d13, d0, d23
fmsub h25, h28, h12, h24
fmsub s3, s5, s6, s31
fmsub d3, d13, d0, d23
fnmadd h3, h18, h31, h24
fnmadd s3, s5, s6, s31
fnmadd d3, d13, d0, d23
fnmsub s3, s5, s6, s31
fnmsub d3, d13, d0, d23
fnmsub h3, h29, h24, h17

#------------------------------------------------------------------------------
# Floating-point <-> fixed-point conversion
#------------------------------------------------------------------------------

fcvtzs  w3, h5, #1
fcvtzs  wzr, h20, #13
fcvtzs  w19, h0, #32
fcvtzs  x3, h5, #1
fcvtzs  x12, h30, #45
fcvtzs  x19, h0, #64
fcvtzs  w3, s5, #1
fcvtzs  wzr, s20, #13
fcvtzs  w19, s0, #32
fcvtzs  x3, s5, #1
fcvtzs  x12, s30, #45
fcvtzs  x19, s0, #64
fcvtzs  w3, d5, #1
fcvtzs  wzr, d20, #13
fcvtzs  w19, d0, #32
fcvtzs  x3, d5, #1
fcvtzs  x12, d30, #45
fcvtzs  x19, d0, #64
fcvtzu  w3, h5, #1
fcvtzu  wzr, h20, #13
fcvtzu  w19, h0, #32
fcvtzu  x3, h5, #1
fcvtzu  x12, h30, #45
fcvtzu  x19, h0, #64
fcvtzu  w3, s5, #1
fcvtzu  wzr, s20, #13
fcvtzu  w19, s0, #32
fcvtzu  x3, s5, #1
fcvtzu  x12, s30, #45
fcvtzu  x19, s0, #64
fcvtzu  w3, d5, #1
fcvtzu  wzr, d20, #13
fcvtzu  w19, d0, #32
fcvtzu  x3, d5, #1
fcvtzu  x12, d30, #45
fcvtzu  x19, d0, #64
scvtf   h23, w19, #1
scvtf   h31, wzr, #20
scvtf   h14, w0, #32
scvtf   h23, x19, #1
scvtf   h31, xzr, #20
scvtf   h14, x0, #64
scvtf   s23, w19, #1
scvtf   s31, wzr, #20
scvtf   s14, w0, #32
scvtf   s23, x19, #1
scvtf   s31, xzr, #20
scvtf   s14, x0, #64
scvtf   d23, w19, #1
scvtf   d31, wzr, #20
scvtf   d14, w0, #32
scvtf   d23, x19, #1
scvtf   d31, xzr, #20
scvtf   d14, x0, #64
ucvtf   h23, w19, #1
ucvtf   h31, wzr, #20
ucvtf   h14, w0, #32
ucvtf   h23, x19, #1
ucvtf   h31, xzr, #20
ucvtf   h14, x0, #64
ucvtf   s23, w19, #1
ucvtf   s31, wzr, #20
ucvtf   s14, w0, #32
ucvtf   s23, x19, #1
ucvtf   s31, xzr, #20
ucvtf   s14, x0, #64
ucvtf   d23, w19, #1
ucvtf   d31, wzr, #20
ucvtf   d14, w0, #32
ucvtf   d23, x19, #1
ucvtf   d31, xzr, #20
ucvtf   d14, x0, #64

#------------------------------------------------------------------------------
# Floating-point <-> integer conversion
#------------------------------------------------------------------------------

fcvtns   w3, h31
fcvtns   xzr, h12
fcvtnu   wzr, h12
fcvtnu   x0, h0
fcvtps   wzr, h9
fcvtps   x12, h20
fcvtpu   w30, h23
fcvtpu   x29, h3
fcvtms   w2, h3
fcvtms   x4, h5
fcvtmu   w6, h7
fcvtmu   x8, h9
fcvtzs   w10, h11
fcvtzs   x12, h13
fcvtzu   w14, h15
fcvtzu   x15, h16
scvtf    h17, w18
scvtf    h19, x20
ucvtf    h21, w22
scvtf    h23, x24
fcvtas   w25, h26
fcvtas   x27, h28
fcvtau   w29, h30
fcvtau   xzr, h0
fcvtns   w3, s31
fcvtns   xzr, s12
fcvtnu   wzr, s12
fcvtnu   x0, s0
fcvtps   wzr, s9
fcvtps   x12, s20
fcvtpu   w30, s23
fcvtpu   x29, s3
fcvtms   w2, s3
fcvtms   x4, s5
fcvtmu   w6, s7
fcvtmu   x8, s9
fcvtzs   w10, s11
fcvtzs   x12, s13
fcvtzu   w14, s15
fcvtzu   x15, s16
scvtf    s17, w18
scvtf    s19, x20
ucvtf    s21, w22
scvtf    s23, x24
fcvtas   w25, s26
fcvtas   x27, s28
fcvtau   w29, s30
fcvtau   xzr, s0
fcvtns   w3, d31
fcvtns   xzr, d12
fcvtnu   wzr, d12
fcvtnu   x0, d0
fcvtps   wzr, d9
fcvtps   x12, d20
fcvtpu   w30, d23
fcvtpu   x29, d3
fcvtms   w2, d3
fcvtms   x4, d5
fcvtmu   w6, d7
fcvtmu   x8, d9
fcvtzs   w10, d11
fcvtzs   x12, d13
fcvtzu   w14, d15
fcvtzu   x15, d16
scvtf    d17, w18
scvtf    d19, x20
ucvtf    d21, w22
ucvtf    d23, x24
fcvtas   w25, d26
fcvtas   x27, d28
fcvtau   w29, d30
fcvtau   xzr, d0
fmov     h6, w5
fmov     h16, x27
fmov     w15, h31
fmov     w3, s9
fmov     s9, w3
fmov     x21, h14
fmov     x20, d31
fmov     d1, x15
fmov     x3, v12.d[1]
fmov     v1.d[1], x19

#------------------------------------------------------------------------------
# Floating-point immediate
#------------------------------------------------------------------------------

fmov     h29, #0.50000000
fmov     s2, #0.12500000
fmov     s3, #1.00000000
fmov     d30, #16.00000000
fmov     s4, #1.06250000
fmov     d10, #1.93750000
fmov     s12, #-1.00000000
fmov     d16, #8.50000000

#------------------------------------------------------------------------------
# Load-register (literal)
#------------------------------------------------------------------------------

ldr       w3, #0
ldr       x29, #4
ldrsw     xzr, #-4
ldr       s0, #8
ldr       d0, #1048572
ldr       q0, #-1048576
prfm      pldl1strm, #0
prfm      #22, #0

#------------------------------------------------------------------------------
# Load/store exclusive
#------------------------------------------------------------------------------

stxrb      w18, w8, [sp]
stxrh      w24, w15, [x16]
stxr       w5, w6, [x17]
stxr       w1, x10, [x21]
ldxrb      w30, [x0]
ldxrh      w17, [x4]
ldxr       w22, [sp]
ldxr       x11, [x29]
stxp       w12, w11, w10, [sp]
stxp       wzr, x27, x9, [x12]
ldxp       w0, wzr, [sp]
ldxp       x17, x0, [x18]
stlxrb     w12, w22, [x0]
stlxrh     w10, w1, [x1]
stlxr      w9, w2, [x2]
stlxr      w9, x3, [sp]
ldaxrb     w8, [x4]
ldaxrh     w7, [x5]
ldaxr      w6, [sp]
ldaxr      x5, [x6]
stlxp      w4, w5, w6, [sp]
stlxp      wzr, x6, x7, [x1]
ldaxp      w5, w18, [sp]
ldaxp      x6, x19, [x22]
stlrb      w24, [sp]
stlrh      w25, [x30]
stlr       w26, [x29]
stlr       x27, [x28]
ldarb      w16, [x21]
ldarb      w23, [sp]
ldarh      w22, [x30]
ldar       wzr, [x29]
ldar       x21, [x28]

#------------------------------------------------------------------------------
# Load/store (unscaled  immediate)
#------------------------------------------------------------------------------

sturb    w9, [sp]
sturh    wzr, [x12, #255]
stur     w16, [x0, #-256]
stur     x28, [x14, #1]
ldurb    w1, [x20, #255]
ldurh    w20, [x1, #255]
ldur     w12, [sp, #255]
ldur     xzr, [x12, #255]
ldursb   x9, [x7, #-256]
ldursh   x17, [x19, #-256]
ldursw   x20, [x15, #-256]
prfum    pldl2keep, [sp, #-256]
ldursb   w19, [x1, #-256]
ldursh   w15, [x21, #-256]
stur     b0, [sp, #1]
stur     h12, [x12, #-1]
stur     s15, [x0, #255]
stur     d31, [x5, #25]
stur     q9, [x5]
ldur     b3, [sp]
ldur     h5, [x4, #-256]
ldur     s7, [x12, #-1]
ldur     d11, [x19, #4]
ldur     q13, [x1, #2]

#------------------------------------------------------------------------------
# Load/store (immediate post-indexed)
#------------------------------------------------------------------------------

strb     w9, [x2], #255
strb     w10, [x3], #1
strb     w10, [x3], #-256
strh     w9, [x2], #255
strh     w9, [x2], #1
strh     w10, [x3], #-256
str      w19, [sp], #255
str      w20, [x30], #1
str      w21, [x12], #-256
str      xzr, [x9], #255
str      x2, [x3], #1
str      x19, [x12], #-256
ldrb     w9, [x2], #255
ldrb     w10, [x3], #1
ldrb     w10, [x3], #-256
ldrh     w9, [x2], #255
ldrh     w9, [x2], #1
ldrh     w10, [x3], #-256
ldr      w19, [sp], #255
ldr      w20, [x30], #1
ldr      w21, [x12], #-256
ldr      xzr, [x9], #255
ldr      x2, [x3], #1
ldr      x19, [x12], #-256
ldrsb    xzr, [x9], #255
ldrsb    x2, [x3], #1
ldrsb    x19, [x12], #-256
ldrsh    xzr, [x9], #255
ldrsh    x2, [x3], #1
ldrsh    x19, [x12], #-256
ldrsw    xzr, [x9], #255
ldrsw    x2, [x3], #1
ldrsw    x19, [x12], #-256
ldrsb    wzr, [x9], #255
ldrsb    w2, [x3], #1
ldrsb    w19, [x12], #-256
ldrsh    wzr, [x9], #255
ldrsh    w2, [x3], #1
ldrsh    w19, [x12], #-256
str      b0, [x0], #255
str      b3, [x3], #1
str      b5, [sp], #-256
str      h10, [x10], #255
str      h13, [x23], #1
str      h15, [sp], #-256
str      s20, [x20], #255
str      s23, [x23], #1
str      s25, [x0], #-256
str      d20, [x20], #255
str      d23, [x23], #1
str      d25, [x0], #-256
ldr      b0, [x0], #255
ldr      b3, [x3], #1
ldr      b5, [sp], #-256
ldr      h10, [x10], #255
ldr      h13, [x23], #1
ldr      h15, [sp], #-256
ldr      s20, [x20], #255
ldr      s23, [x23], #1
ldr      s25, [x0], #-256
ldr      d20, [x20], #255
ldr      d23, [x23], #1
ldr      d25, [x0], #-256
ldr      q20, [x1], #255
ldr      q23, [x9], #1
ldr      q25, [x20], #-256
str      q10, [x1], #255
str      q22, [sp], #1
str      q21, [x20], #-256

#-------------------------------------------------------------------------------
# Load-store register (immediate pre-indexed)
#-------------------------------------------------------------------------------

ldr      x3, [x4, #0]!
strb     w9, [x2, #255]!
strb     w10, [x3, #1]!
strb     w10, [x3, #-256]!
strh     w9, [x2, #255]!
strh     w9, [x2, #1]!
strh     w10, [x3, #-256]!
str      w19, [sp, #255]!
str      w20, [x30, #1]!
str      w21, [x12, #-256]!
str      xzr, [x9, #255]!
str      x2, [x3, #1]!
str      x19, [x12, #-256]!
ldrb     w9, [x2, #255]!
ldrb     w10, [x3, #1]!
ldrb     w10, [x3, #-256]!
ldrh     w9, [x2, #255]!
ldrh     w9, [x2, #1]!
ldrh     w10, [x3, #-256]!
ldr      w19, [sp, #255]!
ldr      w20, [x30, #1]!
ldr      w21, [x12, #-256]!
ldr      xzr, [x9, #255]!
ldr      x2, [x3, #1]!
ldr      x19, [x12, #-256]!
ldrsb    xzr, [x9, #255]!
ldrsb    x2, [x3, #1]!
ldrsb    x19, [x12, #-256]!
ldrsh    xzr, [x9, #255]!
ldrsh    x2, [x3, #1]!
ldrsh    x19, [x12, #-256]!
ldrsw    xzr, [x9, #255]!
ldrsw    x2, [x3, #1]!
ldrsw    x19, [x12, #-256]!
ldrsb    wzr, [x9, #255]!
ldrsb    w2, [x3, #1]!
ldrsb    w19, [x12, #-256]!
ldrsh    wzr, [x9, #255]!
ldrsh    w2, [x3, #1]!
ldrsh    w19, [x12, #-256]!
str      b0, [x0, #255]!
str      b3, [x3, #1]!
str      b5, [sp, #-256]!
str      h10, [x10, #255]!
str      h13, [x23, #1]!
str      h15, [sp, #-256]!
str      s20, [x20, #255]!
str      s23, [x23, #1]!
str      s25, [x0, #-256]!
str      d20, [x20, #255]!
str      d23, [x23, #1]!
str      d25, [x0, #-256]!
ldr      b0, [x0, #255]!
ldr      b3, [x3, #1]!
ldr      b5, [sp, #-256]!
ldr      h10, [x10, #255]!
ldr      h13, [x23, #1]!
ldr      h15, [sp, #-256]!
ldr      s20, [x20, #255]!
ldr      s23, [x23, #1]!
ldr      s25, [x0, #-256]!
ldr      d20, [x20, #255]!
ldr      d23, [x23, #1]!
ldr      d25, [x0, #-256]!
ldr      q20, [x1, #255]!
ldr      q23, [x9, #1]!
ldr      q25, [x20, #-256]!
str      q10, [x1, #255]!
str      q22, [sp, #1]!
str      q21, [x20, #-256]!

#------------------------------------------------------------------------------
# Load/store (unprivileged)
#------------------------------------------------------------------------------

sttrb    w9, [sp]
sttrh    wzr, [x12, #255]
sttr     w16, [x0, #-256]
sttr     x28, [x14, #1]
ldtrb    w1, [x20, #255]
ldtrh    w20, [x1, #255]
ldtr     w12, [sp, #255]
ldtr     xzr, [x12, #255]
ldtrsb   x9, [x7, #-256]
ldtrsh   x17, [x19, #-256]
ldtrsw   x20, [x15, #-256]
ldtrsb   w19, [x1, #-256]
ldtrsh   w15, [x21, #-256]

#------------------------------------------------------------------------------
# Load/store (unsigned  immediate)
#------------------------------------------------------------------------------

ldr      x4, [x29]
ldr      x30, [x12, #32760]
ldr      x20, [sp, #8]
ldr      xzr, [sp]
ldr      w2, [sp]
ldr      w17, [sp, #16380]
ldr      w13, [x2, #4]
ldrsw    x2, [x5, #4]
ldrsw    x23, [sp, #16380]
ldrsw    x21, [x25, x7]
ldrh     w2, [x4]
ldrsh    w23, [x6, #8190]
ldrsh    wzr, [sp, #2]
ldrsh    x29, [x2, #2]
ldrsh    x25, [x8, w13, uxtw]
ldrb     w26, [x3, #121]
ldrb     w12, [x2]
ldrsb    w27, [sp, #4095]
ldrsb    xzr, [x15]
ldrsb    x12, [x28, x27]
str      x30, [sp]
str      w20, [x4, #16380]
str      b5, [x11]
str      h23, [x15]
str      s25, [x19]
str      d15, [x2]
strh     w17, [sp, #8190]
strb     w23, [x3, #4095]
strb     wzr, [x2]
ldr      b31, [sp, #4095]
ldr      h20, [x2, #8190]
ldr      s10, [x19, #16380]
ldr      d3, [x10, #32760]
str      q12, [sp, #65520]
ldr      q14, [x6, #4624]

#------------------------------------------------------------------------------
# Load/store (register offset)
#------------------------------------------------------------------------------

ldrb     w3, [sp, x5]
ldrb     w9, [x27, x6]
ldrsb    w10, [x30, x7]
ldrb     w11, [x29, x3, sxtx]
strb     w12, [x28, xzr, sxtx]
strb     w5, [x26, w7, uxtw]
ldrb     w14, [x26, w6, uxtw]
ldrsb    w15, [x25, w7, uxtw]
ldrb     w17, [x23, w9, sxtw]
ldrsb    x18, [x22, w10, sxtw]
ldrsh    w3, [sp, x5]
ldrsh    w9, [x27, x6]
ldrh     w10, [x30, x7, lsl #1]
strh     w11, [x29, x3, sxtx]
ldrh     w12, [x28, xzr, sxtx]
ldrsh    x13, [x27, x5, sxtx #1]
ldrh     w14, [x26, w6, uxtw]
ldrh     w15, [x25, w7, uxtw]
ldrsh    w16, [x24, w8, uxtw #1]
ldrh     w17, [x23, w9, sxtw]
ldrh     w18, [x22, w10, sxtw]
strh     w19, [x21, wzr, sxtw #1]
ldr      b25, [x21, w8, uxtw]
ldr      b8, [x30, x10]
str      b14, [x13, x25]
str      b30, [x16, w26, uxtw]
ldr      h3, [sp, x5]
ldr      h9, [x27, x6]
ldr      h10, [x30, x7, lsl #1]
str      h11, [x29, x3, sxtx]
str      h12, [x28, xzr, sxtx]
str      h13, [x27, x5, sxtx #1]
ldr      h14, [x26, w6, uxtw]
ldr      h15, [x25, w7, uxtw]
ldr      h16, [x24, w8, uxtw #1]
ldr      h17, [x23, w9, sxtw]
str      h18, [x22, w10, sxtw]
ldr      h19, [x21, wzr, sxtw #1]
ldr      s12, [x30, w5, uxtw]
ldr      d24, [x26, w7, uxtw]
str      s20, [x24, w10, uxtw]
str      d5, [x26, x6]
ldr      w3, [sp, x5]
ldr      s9, [x27, x6]
ldr      w10, [x30, x7, lsl #2]
ldr      w11, [x29, x3, sxtx]
str      s12, [x28, xzr, sxtx]
str      w13, [x27, x5, sxtx #2]
str      w14, [x26, w6, uxtw]
ldr      w15, [x25, w7, uxtw]
ldr      w16, [x24, w8, uxtw #2]
ldrsw    x17, [x23, w9, sxtw]
ldr      w18, [x22, w10, sxtw]
ldrsw    x19, [x21, wzr, sxtw #2]
ldr      x3, [sp, x5]
str      x9, [x27, x6]
ldr      d10, [x30, x7, lsl #3]
str      x11, [x29, x3, sxtx]
ldr      x12, [x28, xzr, sxtx]
ldr      x13, [x27, x5, sxtx #3]
prfm     pldl1keep, [x26, w6, uxtw]
ldr      x15, [x25, w7, uxtw]
str      x27, [x26, w24, uxtw]
ldr      x16, [x24, w8, uxtw #3]
ldr      x17, [x23, w9, sxtw]
ldr      x18, [x22, w10, sxtw]
str      d19, [x21, wzr, sxtw #3]
ldr      q3, [sp, x5]
ldr      q9, [x27, x6]
ldr      q10, [x30, x7, lsl #4]
str      q11, [x29, x3, sxtx]
str      q12, [x28, xzr, sxtx]
str      q13, [x27, x5, sxtx #4]
ldr      q14, [x26, w6, uxtw]
ldr      q15, [x25, w7, uxtw]
ldr      q16, [x24, w8, uxtw #4]
ldr      q17, [x23, w9, sxtw]
str      q18, [x22, w10, sxtw]
ldr      q19, [x21, wzr, sxtw #4]

#------------------------------------------------------------------------------
# Load/store register pair (offset)
#------------------------------------------------------------------------------

ldp      w3, w5, [sp]
stp      wzr, w9, [sp, #252]
ldp      w2, wzr, [sp, #-256]
ldp      w9, w10, [sp, #4]
ldpsw    x9, x10, [sp, #4]
ldpsw    x9, x10, [x2, #-256]
ldpsw    x20, x30, [sp, #252]
ldp      x21, x29, [x2, #504]
ldp      x22, x23, [x3, #-512]
ldp      x24, x25, [x4, #8]
ldp      s29, s28, [sp, #252]
stp      s27, s26, [sp, #-256]
ldp      s1, s2, [x3, #44]
stp      d3, d5, [x9, #504]
stp      d7, d11, [x10, #-512]
stnp     x20, x16, [x8]
stp      x3, x6, [x16]
ldp      d2, d3, [x30, #-8]
stp      q3, q5, [sp]
stp      q17, q19, [sp, #1008]
ldp      q23, q29, [x1, #-1024]

#------------------------------------------------------------------------------
# Load/store register pair (post-indexed)
#------------------------------------------------------------------------------

ldp      w3, w5, [sp], #0
stp      wzr, w9, [sp], #252
ldp      w2, wzr, [sp], #-256
ldp      w9, w10, [sp], #4
ldpsw    x9, x10, [sp], #4
ldpsw    x9, x10, [x2], #-256
ldpsw    x20, x30, [sp], #252
ldp      x21, x29, [x2], #504
ldp      x22, x23, [x3], #-512
ldp      x24, x25, [x4], #8
ldp      s29, s28, [sp], #252
stp      s27, s26, [sp], #-256
ldp      s1, s2, [x3], #44
stp      d3, d5, [x9], #504
stp      d7, d11, [x10], #-512
ldp      d2, d3, [x30], #-8
stp      q3, q5, [sp], #0
stp      q17, q19, [sp], #1008
ldp      q23, q29, [x1], #-1024

#------------------------------------------------------------------------------
# Load/store register pair (pre-indexed)
#------------------------------------------------------------------------------

ldp      w3, w5, [sp, #0]!
stp      wzr, w9, [sp, #252]!
ldp      w2, wzr, [sp, #-256]!
ldp      w9, w10, [sp, #4]!
ldpsw    x9, x10, [sp, #4]!
ldpsw    x9, x10, [x2, #-256]!
ldpsw    x20, x30, [sp, #252]!
ldp      x21, x29, [x2, #504]!
ldp      x22, x23, [x3, #-512]!
ldp      x24, x25, [x4, #8]!
ldp      s29, s28, [sp, #252]!
stp      s27, s26, [sp, #-256]!
ldp      s1, s2, [x3, #44]!
stp      d3, d5, [x9, #504]!
stp      d7, d11, [x10, #-512]!
ldp      d2, d3, [x30, #-8]!
stp      q3, q5, [sp, #0]!
stp      q17, q19, [sp, #1008]!
ldp      q23, q29, [x1, #-1024]!

#------------------------------------------------------------------------------
# Load/store register pair (offset)
#------------------------------------------------------------------------------

ldnp      w3, w5, [sp]
stnp      wzr, w9, [sp, #252]
ldnp      w2, wzr, [sp, #-256]
ldnp      w9, w10, [sp, #4]
ldnp      x21, x29, [x2, #504]
ldnp      x22, x23, [x3, #-512]
ldnp      x24, x25, [x4, #8]
ldnp      s29, s28, [sp, #252]
stnp      s27, s26, [sp, #-256]
ldnp      s1, s2, [x3, #44]
stnp      d3, d5, [x9, #504]
stnp      d7, d11, [x10, #-512]
ldnp      d2, d3, [x30, #-8]
stnp      q3, q5, [sp]
stnp      q17, q19, [sp, #1008]
ldnp      q23, q29, [x1, #-1024]

#------------------------------------------------------------------------------
# Logical (immediate)
#------------------------------------------------------------------------------

and      wsp, w16, #0xe00
and      x2, x22, #0x1e00
ands     w14, w8, #0x70
ands     x4, x10, #0x60
eor      wsp, w4, #0xe00
eor      x27, x25, #0x1e00
mov      w3, #983055
mov      x10, #-6148914691236517206

#------------------------------------------------------------------------------
# Logical (shifted register)
#------------------------------------------------------------------------------

and      w12, w23, w21
and      w16, w15, w1, lsl #1
and      w9, w4, w10, lsl #31
and      w3, w30, w11
and      x3, x5, x7, lsl #63
and      x5, x14, x19, asr #4
and      w3, w17, w19, ror #31
and      w0, w2, wzr, lsr #17
and      w3, w30, w11, asr #2
and      xzr, x4, x26
and      w3, wzr, w20, ror #2
and      x7, x20, xzr, asr #63
bic      x13, x20, x14, lsl #47
bic      w2, w7, w9
eon      w29, w4, w19
eon      x19, x12, x2
eor      w8, w27, w2
eor      x22, x16, x6
orr      w2, w7, w0, asr #31
orr      x8, x9, x10, lsl #12
orn      x3, x5, x7, asr #2
orn      w2, w5, w29
ands     w7, wzr, w9, lsl #1
ands     x3, x5, x20, ror #63
bics     w3, w5, w7
bics     x3, xzr, x3, lsl #1
tst      w3, w7, lsl #31
tst      x2, x20, asr #2
mov      x3, x6
mov      x3, xzr
mov      wzr, w2
mov      w3, w5

#------------------------------------------------------------------------------
# Move wide (immediate)
#------------------------------------------------------------------------------

movz     w2, #0, lsl #16
mov     w2, #-1235
mov     x2, #5299989643264
mov      x2, #0
movk     w3, #0
movz     x4, #0, lsl #16
movk     w5, #0, lsl #16
movz     x6, #0, lsl #32
movk     x7, #0, lsl #32
movz     x8, #0, lsl #48
movk     x9, #0, lsl #48

#------------------------------------------------------------------------------
# Move immediate to Special Register
#------------------------------------------------------------------------------

msr     DAIFSet, #0

#------------------------------------------------------------------------------
# PC-relative addressing
#------------------------------------------------------------------------------

adr      x2, #1600
adrp     x21, #6553600
adr      x0, #262144

#------------------------------------------------------------------------------
# Test and branch (immediate)
#------------------------------------------------------------------------------

tbz     x12, #62, #0
tbz     x12, #62, #4
tbz     x12, #62, #-32768
tbz     w17, #16, test
tbnz    x12, #60, #32764
tbnz	w3, #28, test

#------------------------------------------------------------------------------
# Unconditional branch (immediate)
#------------------------------------------------------------------------------

b        #4
b        #-4
b        #134217724
bl       test

#------------------------------------------------------------------------------
# Unconditional branch (register)
#------------------------------------------------------------------------------

br       x20
blr      xzr
ret      x10
ret
eret
drps
