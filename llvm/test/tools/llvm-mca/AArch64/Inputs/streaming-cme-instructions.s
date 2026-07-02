// Instructions available with no SME but will be sent to CME when in streaming
// SVE mode

fadd s0, s1, s2
fcvtzs w0, s1
frintz s0, s1
scvtf s0, w0
fmov d0, x0
fnmadd s0, s1, s2, s3
ldr s0, [x0]
ldr q0, [x0, x1, lsl #4]
str s0, [x0]
fadd v0.4s, v1.4s, v2.4s
bfdot v0.4s, v1.8h, v2.8h
add z0.s, z1.s, z2.s
andv b0, p7, z31.b
clastb w0, p0, w0, z2.s
cpy z0.s, p0/m, w0
dup z0.s, w0
fdup z0.h, #1.00000
ext z0.b, z0.b, z1.b, #1
tbl z0.b, { z0.b, z1.b }, z2.b
fadd z0.s, p0/m, z0.s, z1.s
fmul z0.s, p0/m, z0.s, z1.s
fmin z0.s, p0/m, z0.s, z1.s
fcpy z0.s, p0/m, #1.0
bfdot z0.s, z1.h, z2.h
ld1w { z0.s }, p0/z, [x0]
ld2w { z0.s, z1.s }, p0/z, [x0]
ld1d { z0.d }, p0/z, [x0, z0.d, uxtw]
st1w { z0.s }, p0, [x0]
st2w { z0.s, z1.s }, p0, [x0]
st1w { z0.s }, p0, [x0, z0.s, uxtw]
eor3 z0.d, z0.d, z1.d, z2.d
sabalb z0.s, z1.h, z2.h
sadalp z0.d, p0/m, z1.s
ssra z0.d, z1.d, #1
mla z0.d, p0/m, z1.d, z2.d
index z0.s, w0, #1
index z0.s, #0, w0
index z0.s, w0, w1
insr z0.s, w0
lasta d0, p7, z31.d
faddv d0, p0, z1.d
fmaxnmv d0, p0, z1.d
sqcvtn z0.h, { z0.s, z1.s }
uaddv d0, p7, z31.b
