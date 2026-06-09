// Instructions only available when in streaming SVE mode
// and sent to CME.

// Table 2-43: SVE2 and base A64 instructions added by SME.
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
sqrshrn z0.b, {z0.s - z3.s}, #32
sqrshrun z0.b, {z0.s - z3.s}, #32
uqrshrn z0.b, {z0.s - z3.s}, #32

// Table 2-44: SME and SME2 processing instructions.
add za.s[w8, 0, vgx2], {z0.s, z1.s}
sub za.s[w8, 0, vgx2], {z0.s, z1.s}
add {z0.h, z1.h}, {z0.h, z1.h}, z0.h
addha za0.s, p0/m, p0/m, z0.s
addva za0.s, p0/m, p0/m, z0.s
bf1cvt z0.h, z0.b
bf1cvtlt z0.h, z0.b
bf2cvt z0.h, z0.b
bf2cvtlt z0.h, z0.b
bfadd za.h[w8, 0, vgx2], {z0.h, z1.h}
bfcvt z0.h, {z0.s, z1.s}
bfcvtn z0.h, {z0.s, z1.s}
bfclamp {z0.h, z1.h}, z0.h, z0.h
bfdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h
bfmax {z0.h, z1.h}, {z0.h, z1.h}, z0.h
bfmaxnm {z0.h, z1.h}, {z0.h, z1.h}, z0.h
bfmin {z0.h, z1.h}, {z0.h, z1.h}, z0.h
bfminnm {z0.h, z1.h}, {z0.h, z1.h}, z0.h
bfmla za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h
bfmlal za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h
bfmls za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h
bfmlsl za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h
bfmopa za0.s, p0/m, p0/m, z0.h, z0.h
bfmops za0.s, p0/m, p0/m, z0.h, z0.h
bfvdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]
bmopa za0.s, p0/m, p0/m, z0.s, z0.s
bmops za0.s, p0/m, p0/m, z0.s, z0.s
f1cvt z0.h, z0.b
f1cvtlt z0.h, z0.b
f2cvt z0.h, z0.b
f2cvtlt z0.h, z0.b
fadd za.s[w8, 0, vgx2], {z0.s, z1.s}
famax {z0.s, z1.s}, {z0.s, z1.s}, {z0.s, z1.s}
famin {z0.s, z1.s}, {z0.s, z1.s}, {z0.s, z1.s}
fclamp {z0.h, z1.h}, z0.h, z0.h
fcvt z0.h, {z0.s, z1.s}
fcvtn z0.h, {z0.s, z1.s}
fcvtzs {z0.s, z1.s}, {z0.s, z1.s}
fcvtzu {z0.s, z1.s}, {z0.s, z1.s}
fmopa za0.s, p0/m, p0/m, z0.h, z0.h
fmops za0.s, p0/m, p0/m, z0.h, z0.h
fmax {z0.s, z1.s}, {z0.s, z1.s}, z0.s
fmaxnm {z0.s, z1.s}, {z0.s, z1.s}, z0.s
fmin {z0.s, z1.s}, {z0.s, z1.s}, z0.s
fminnm {z0.s, z1.s}, {z0.s, z1.s}, z0.s
fmla za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s
fmlal za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h
fmlall za.s[w8, 0:3], z0.b, z0.b
fmls za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s
fmlsl za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h
frinta {z0.s, z1.s}, {z0.s, z1.s}
frintm {z0.s, z1.s}, {z0.s, z1.s}
frintn {z0.s, z1.s}, {z0.s, z1.s}
frintp {z0.s, z1.s}, {z0.s, z1.s}
fvdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]
fvdotb za.s[w8, 0, vgx4], {z0.b, z1.b}, z0.b[0]
fvdott za.s[w8, 0, vgx4], {z0.b, z1.b}, z0.b[0]
fscale {z0.s, z1.s}, {z0.s, z1.s}, z0.s
luti2 {z0.h - z1.h}, zt0, z0[0]
luti4 {z0.h - z1.h}, zt0, z0[0]
mova {z0.s, z1.s}, za0h.s[w12, 0:1]
movt zt0[0], x0
movt x0, zt0[0]
sclamp {z0.h, z1.h}, z0.h, z0.h
smax {z0.s, z1.s}, {z0.s, z1.s}, z0.s
smin {z0.s, z1.s}, {z0.s, z1.s}, z0.s
uclamp {z0.h, z1.h}, z0.h, z0.h
umax {z0.s, z1.s}, {z0.s, z1.s}, z0.s
umin {z0.s, z1.s}, {z0.s, z1.s}, z0.s
scvtf {z0.s, z1.s}, {z0.s, z1.s}
ucvtf {z0.s, z1.s}, {z0.s, z1.s}
sel {z0.h, z1.h}, pn8, {z0.h, z1.h}, {z0.h, z1.h}
sel {z0.h, z1.h, z2.h, z3.h}, pn8, {z0.h, z1.h, z2.h, z3.h}, {z0.h, z1.h, z2.h, z3.h}
smlal za.s[w8, 0:1], z0.h, z0.h
smlall za.s[w8, 0:3], z0.b, z0.b
smlsl za.s[w8, 0:1], z0.h, z0.h
smlsll za.s[w8, 0:3], z0.b, z0.b
umlal za.s[w8, 0:1], z0.h, z0.h
umlall za.s[w8, 0:3], z0.b, z0.b
umlsl za.s[w8, 0:1], z0.h, z0.h
umlsll za.s[w8, 0:3], z0.b, z0.b
smopa za0.s, p0/m, p0/m, z0.h, z0.h
smops za0.s, p0/m, p0/m, z0.h, z0.h
umopa za0.s, p0/m, p0/m, z0.h, z0.h
umops za0.s, p0/m, p0/m, z0.h, z0.h
sqcvtn z0.h, {z0.s, z1.s}
sqcvtun z0.b, {z0.s - z3.s}
uqcvtn z0.h, {z0.s, z1.s}
sqdmulh {z0.h - z1.h}, {z0.h - z1.h}, z0.h
sqrshr z0.h, {z0.s - z1.s}, #16
sqrshrn z0.b, {z0.s - z3.s}, #32
sqrshru z0.h, {z0.s - z1.s}, #16
sqrshrun z0.b, {z0.s - z3.s}, #32
uqrshr z0.h, {z0.s - z1.s}, #16
uqrshrn z0.b, {z0.s - z3.s}, #32
srshl {z0.h, z1.h}, {z0.h, z1.h}, z0.h
urshl {z0.h, z1.h}, {z0.h, z1.h}, z0.h
sudot za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b
usdot za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b
usmlall za.s[w8, 0:3], z0.b, z0.b
sumopa za0.s, p0/m, p0/m, z0.b, z0.b
sumops za0.s, p0/m, p0/m, z0.b, z0.b
usmopa za0.s, p0/m, p0/m, z0.b, z0.b
usmops za0.s, p0/m, p0/m, z0.b, z0.b
sunpk {z0.h - z1.h}, z0.b
uunpk {z0.h - z1.h}, z0.b
suvdot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]
uvdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]
svdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]
usvdot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]
uzp1 z31.s, z31.s, z31.s
uzp2 z31.s, z31.s, z31.s
zero za.d[w8, 0, vgx2]
zero {zt0}
zip1 z0.s, z0.s, z0.s
zip2 z0.s, z0.s, z0.s

// Table 2-45: SME load instructions.
ldr za[w12, #0], [x0]

// Table 2-46: SME store instructions.
str za[w12, #0], [x0]
