// SVE instructions added by SME and available when not in Streaming SVE mode

bfmlslb z0.s, z1.h, z2.h
fdot z0.s, z1.h, z2.h
psel p0, p0, p0.b[w12, 0]
