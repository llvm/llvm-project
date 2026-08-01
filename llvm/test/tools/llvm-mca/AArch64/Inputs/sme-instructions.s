// Instructions added by SME and available when not in streaming. When in
// streaming these are not sent to CME
addspl x0, x1, #10
addspl x0, sp, #10
addsvl x0, x1, #10
addsvl x0, sp, #10
rdsvl  x1, #10
cntp x0, pn0.h, vlx2
pext p0.h, pn8[0]
pext { p0.h, p1.h }, pn8[0]
ptrue pn8.h
whilege pn8.h, x0, x0, vlx2
whilegt pn8.h, x0, x0, vlx2
whilehi pn8.h, x0, x0, vlx2
whilehs pn8.h, x0, x0, vlx2
whilele pn8.h, x0, x0, vlx2
whilelo pn8.h, x0, x0, vlx2
whilels pn8.h, x0, x0, vlx2
whilelt pn8.h, x0, x0, vlx2
bfmlslb z0.s, z1.h, z2.h
bfmlslb z0.s, z1.h, z2.h[0]
bfmlslt z0.s, z1.h, z2.h
bfmlslt z0.s, z1.h, z2.h[0]
fclamp z0.s, z1.s, z2.s
fdot z0.s, z1.h, z2.h
fdot z0.s, z1.h, z2.h[0]
psel p0, p0, p0.b[w12, 0]
revd z0.q, p0/m, z0.q
sclamp z0.s, z1.s, z2.s
uclamp z0.s, z1.s, z2.s
sdot z0.s, z0.h, z0.h
sdot z0.s, z0.h, z0.h[0]
udot z0.s, z0.h, z0.h
udot z0.s, z0.h, z0.h[0]
sqcvtn z0.h, { z0.s, z1.s }
sqcvtun z0.h, { z0.s, z1.s }
uqcvtn z0.h, { z0.s, z1.s }
whilege { p0.h, p1.h }, x0, x0
whilegt { p0.h, p1.h }, x0, x0
whilehi { p0.h, p1.h }, x0, x0
whilehs { p0.h, p1.h }, x0, x0
whilele { p0.h, p1.h }, x0, x0
whilelo { p0.h, p1.h }, x0, x0
whilels { p0.h, p1.h }, x0, x0
whilelt { p0.h, p1.h }, x0, x0

// SVE2 and base A64 instructions added by SME and available when not in
// streaming. When in streaming these are sent to the CME.
bfmlslb z0.s, z1.h, z2.h
bfmlslt z0.s, z1.h, z2.h
fclamp z0.s, z1.s, z2.s
fdot z0.s, z1.h, z2.h
revd z0.q, p0/m, z0.q
rprfm #0, x0, [x0]
sclamp z0.s, z1.s, z2.s
uclamp z0.s, z1.s, z2.s
sdot z0.s, z0.h, z0.h
udot z0.s, z0.h, z0.h
sqcvtn z0.h, {z0.s, z1.s}
sqcvtun z0.b, {z0.s - z3.s}
uqcvtn z0.h, {z0.s, z1.s}
sqrshrn z0.h, {z0.s - z1.s}, #16
uqrshrn z0.h, {z0.s - z1.s}, #16
sqrshrn z0.b, {z0.s - z3.s}, #32
sqrshrun z0.b, {z0.s - z3.s}, #32
uqrshrn z0.b, {z0.s - z3.s}, #32
