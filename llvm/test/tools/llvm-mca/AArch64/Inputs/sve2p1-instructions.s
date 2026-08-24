// SVE instructions added by SME and available when not in Streaming SVE mode

bfmlslb z0.s, z1.h, z2.h
bfmlslt z0.s, z1.h, z2.h
fclamp z0.s, z1.s, z2.s
fdot z0.s, z1.h, z2.h
psel p0, p0, p0.b[w12, 0]
revd z0.q, p0/m, z0.q
sclamp z0.s, z1.s, z2.s
uclamp z0.s, z1.s, z2.s
sdot z0.s, z0.h, z0.h
udot z0.s, z0.h, z0.h
sqcvtun z0.h, { z0.s, z1.s }
sqrshrun z0.h, { z0.s, z1.s }, #16
