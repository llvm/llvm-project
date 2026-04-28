fadd s0, s1, s2
fcvtzs w0, s1
ldr s0, [x0]
ldr q0, [x0, x1, lsl #4]
str s0, [x0]
fadd v0.4s, v1.4s, v2.4s
bfdot v0.4s, v1.8h, v2.8h
add z0.s, z1.s, z2.s
clastb w0, p0, w0, z2.s
fadd z0.s, p0/m, z0.s, z1.s
bfdot z0.s, z1.h, z2.h
ld1w { z0.s }, p0/z, [x0]
ld2w { z0.s, z1.s }, p0/z, [x0]
ld1d { z0.d }, p0/z, [x0, z0.d, uxtw]
st1w { z0.s }, p0, [x0]
st2w { z0.s, z1.s }, p0, [x0]
st1w { z0.s }, p0, [x0, z0.s, uxtw]
eor3 z0.d, z0.d, z1.d, z2.d
bfmlslb z0.s, z1.h, z2.h
fdot z0.s, z1.h, z2.h
psel p0, p0, p0.b[w12, 0]
sqcvtn z0.h, { z0.s, z1.s }
