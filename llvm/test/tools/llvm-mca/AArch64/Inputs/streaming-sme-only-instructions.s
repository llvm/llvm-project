// Instructions only available when in streaming SVE mode
// and sent to CME.

// SME and SME2 processing instructions.
add za.s[w8, 0, vgx2], {z0.s, z1.s}
sub za.s[w8, 0, vgx2], {z0.s, z1.s}
sub za.s[w8, 0, vgx4], {z0.s - z3.s}
add {z0.h, z1.h}, {z0.h, z1.h}, z0.h
add {z0.h - z3.h}, {z0.h - z3.h}, z0.h
add {z0.s - z3.s}, {z0.s - z3.s}, z4.s
addha za0.s, p0/m, p0/m, z0.s
addva za0.s, p0/m, p0/m, z0.s
bf1cvt {z0.h - z1.h}, z0.b
bf1cvtlt z0.h, z0.b
bf2cvt {z0.h - z1.h}, z0.b
bf2cvtlt z0.h, z0.b
bfadd za.h[w8, 0, vgx2], {z0.h, z1.h}
bfadd za.h[w8, 0, vgx4], {z0.h - z3.h}
bfsub za.h[w8, 0, vgx2], {z0.h, z1.h}
bfcvt z0.h, {z0.s, z1.s}
bfcvtn z0.h, {z0.s, z1.s}
bfclamp {z0.h, z1.h}, z0.h, z0.h
bfclamp {z0.h - z3.h}, z0.h, z0.h
bfdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h
bfdot za.s[w8, 0, vgx4], {z0.h - z3.h}, z0.h
fdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h
fdot za.s[w8, 0, vgx4], {z0.h - z3.h}, z0.h
bfmax {z0.h, z1.h}, {z0.h, z1.h}, z0.h
bfmax {z0.h - z3.h}, {z0.h - z3.h}, z0.h
bfmaxnm {z0.h, z1.h}, {z0.h, z1.h}, z0.h
bfmaxnm {z0.h - z3.h}, {z0.h - z3.h}, z0.h
bfmin {z0.h, z1.h}, {z0.h, z1.h}, z0.h
bfmin {z0.h - z3.h}, {z0.h - z3.h}, z0.h
bfminnm {z0.h, z1.h}, {z0.h, z1.h}, z0.h
bfminnm {z0.h - z3.h}, {z0.h - z3.h}, z0.h
bfmla za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h
bfmla za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h
bfmlal za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h
bfmlal za.s[w8, 0:1], z1.h, z0.h
bfmlal za.s[w8, 0:1], z1.h, z0.h[0]
bfmlal za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h
fmlal za.s[w8, 0:1], z1.h, z0.h
fmlal za.s[w8, 0:1], z1.h, z0.h[0]
bfmls za.h[w8, 0, vgx2], {z0.h, z1.h}, z0.h
bfmls za.h[w8, 0, vgx4], {z0.h - z3.h}, z0.h
bfmlsl za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h
bfmlsl za.s[w8, 0:1], z1.h, z0.h
bfmlsl za.s[w8, 0:1], z1.h, z0.h[0]
bfmlsl za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h
fmlsl za.s[w8, 0:1], z1.h, z0.h
fmlsl za.s[w8, 0:1], z1.h, z0.h[0]
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
fadd za.d[w8, 0, vgx4], {z0.d - z3.d}
famax {z0.s, z1.s}, {z0.s, z1.s}, {z0.s, z1.s}
famin {z0.s, z1.s}, {z0.s, z1.s}, {z0.s, z1.s}
fclamp {z0.h, z1.h}, z0.h, z0.h
fclamp {z0.d - z3.d}, z0.d, z0.d
fcvt z0.b, {z0.s - z3.s}
fcvt z0.h, {z0.s, z1.s}
fcvtnb z0.b, {z0.s, z1.s}
bfcvtn z0.b, {z0.h, z1.h}
fcvtn z0.h, {z0.s, z1.s}
fcvtzs {z0.s, z1.s}, {z0.s, z1.s}
fcvtzs {z0.s - z3.s}, {z0.s - z3.s}
fcvtzu {z0.s, z1.s}, {z0.s, z1.s}
fcvtzu {z0.s - z3.s}, {z0.s - z3.s}
fmopa za0.s, p0/m, p0/m, z0.h, z0.h
fmops za0.s, p0/m, p0/m, z0.h, z0.h
fmax {z0.s, z1.s}, {z0.s, z1.s}, z0.s
fmax {z0.s - z3.s}, {z0.s - z3.s}, z0.s
fmaxnm {z0.s, z1.s}, {z0.s, z1.s}, z0.s
fmaxnm {z0.s - z3.s}, {z0.s - z3.s}, z0.s
fmin {z0.s, z1.s}, {z0.s, z1.s}, z0.s
fmin {z0.s - z3.s}, {z0.s - z3.s}, z0.s
fminnm {z0.s, z1.s}, {z0.s, z1.s}, z0.s
fminnm {z0.s - z3.s}, {z0.s - z3.s}, z0.s
fmla za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s
fmla za.s[w8, 0, vgx4], {z0.s - z3.s}, z0.s
fmlal za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h
fmlal za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h
fmlall za.s[w8, 0:3], z0.b, z0.b
fmls za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s
fmls za.s[w8, 0, vgx4], {z0.s - z3.s}, z0.s
fmlsl za.s[w8, 0:1, vgx2], {z0.h, z1.h}, z0.h
fmlsl za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h
frinta {z0.s, z1.s}, {z0.s, z1.s}
frinta {z0.s - z3.s}, {z0.s - z3.s}
frintm {z0.s, z1.s}, {z0.s, z1.s}
frintm {z0.s - z3.s}, {z0.s - z3.s}
frintn {z0.s, z1.s}, {z0.s, z1.s}
frintn {z0.s - z3.s}, {z0.s - z3.s}
frintp {z0.s, z1.s}, {z0.s, z1.s}
frintp {z0.s - z3.s}, {z0.s - z3.s}
fvdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]
fvdotb za.s[w8, 0, vgx4], {z0.b, z1.b}, z0.b[0]
fvdott za.s[w8, 0, vgx4], {z0.b, z1.b}, z0.b[0]
fscale {z0.s, z1.s}, {z0.s, z1.s}, z0.s
fscale {z0.s - z3.s}, {z0.s - z3.s}, z0.s
luti2 {z0.h - z1.h}, zt0, z0[0]
luti2 {z0.h - z3.h}, zt0, z0[0]
luti4 {z0.h - z1.h}, zt0, z0[0]
luti4 {z0.h - z3.h}, zt0, z0[0]
ld1b {z0.b, z1.b}, pn8/z, [x0]
ld1b {z0.b - z3.b}, pn8/z, [x0]
ldnt1b {z0.b, z1.b}, pn8/z, [x0]
ldnt1b {z0.b - z3.b}, pn8/z, [x0]
mova {z0.s, z1.s}, za0h.s[w12, 0:1]
mova z0.s, p0/m, za0h.s[w12, 0]
mova za0h.b[w12, 0], p0/m, z0.b
mova {z0.h - z3.h}, za0h.h[w12, 0:3]
movt zt0[0], x0
movt x0, zt0[0]
sclamp {z0.h, z1.h}, z0.h, z0.h
sclamp {z0.s - z3.s}, z4.s, z5.s
smax {z0.s, z1.s}, {z0.s, z1.s}, z0.s
smin {z0.s, z1.s}, {z0.s, z1.s}, z0.s
smin {z0.s - z3.s}, {z0.s - z3.s}, z0.s
uclamp {z0.h, z1.h}, z0.h, z0.h
uclamp {z0.s - z3.s}, z4.s, z5.s
umax {z0.s, z1.s}, {z0.s, z1.s}, z0.s
umax {z0.s - z3.s}, {z0.s - z3.s}, z0.s
umin {z0.s, z1.s}, {z0.s, z1.s}, z0.s
smax {z0.s - z3.s}, {z0.s - z3.s}, z0.s
umin {z0.s - z3.s}, {z0.s - z3.s}, z0.s
scvtf {z0.s, z1.s}, {z0.s, z1.s}
scvtf {z0.s - z3.s}, {z0.s - z3.s}
ucvtf {z0.s, z1.s}, {z0.s, z1.s}
ucvtf {z0.s - z3.s}, {z0.s - z3.s}
sel {z0.h, z1.h}, pn8, {z0.h, z1.h}, {z0.h, z1.h}
sel {z0.h, z1.h, z2.h, z3.h}, pn8, {z0.h, z1.h, z2.h, z3.h}, {z0.h, z1.h, z2.h, z3.h}
smlal za.s[w8, 0:1], z0.h, z0.h
smlal za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h
smlall za.s[w8, 0:3], z0.b, z0.b
smlall za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b
sumlall za.s[w11, 4:7, vgx2], {z0.b - z1.b}, z8.b[3]
sumlall za.s[w11, 4:7], z0.b, z8.b[3]
sumlall za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b
smlsl za.s[w8, 0:1], z0.h, z0.h
smlsl za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h
smlsll za.s[w8, 0:3], z0.b, z0.b
smlsll za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b
umlal za.s[w8, 0:1], z0.h, z0.h
umlal za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h
umlall za.s[w8, 0:3], z0.b, z0.b
umlall za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b
umlsl za.s[w8, 0:1], z0.h, z0.h
umlsl za.s[w8, 0:1, vgx4], {z0.h - z3.h}, z0.h
umlsll za.s[w8, 0:3], z0.b, z0.b
umlsll za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b
smopa za0.s, p0/m, p0/m, z0.h, z0.h
smops za0.s, p0/m, p0/m, z0.h, z0.h
umopa za0.s, p0/m, p0/m, z0.h, z0.h
umops za0.s, p0/m, p0/m, z0.h, z0.h
sqcvt z0.h, {z0.s - z1.s}
sqcvt z0.b, {z0.s - z3.s}
sqcvtun z0.b, {z0.s - z3.s}
sqcvtu z0.h, {z0.s, z1.s}
uqcvt z0.h, {z0.s, z1.s}
sqdmulh {z0.h - z1.h}, {z0.h - z1.h}, z0.h
sqdmulh {z0.s - z3.s}, {z0.s - z3.s}, z0.s
sqrshr z0.h, {z0.s - z1.s}, #16
sqrshr z0.b, {z0.s - z3.s}, #32
sqrshrn z0.b, {z0.s - z3.s}, #32
sqrshru z0.h, {z0.s - z1.s}, #16
sqrshru z0.b, {z0.s - z3.s}, #32
sqrshrun z0.b, {z0.s - z3.s}, #32
uqrshr z0.h, {z0.s - z1.s}, #16
uqrshr z0.b, {z0.s - z3.s}, #32
uqrshrn z0.b, {z0.s - z3.s}, #32
srshl {z0.h, z1.h}, {z0.h, z1.h}, z0.h
srshl {z0.s - z3.s}, {z0.s - z3.s}, z0.s
urshl {z0.h, z1.h}, {z0.h, z1.h}, z0.h
urshl {z0.s - z3.s}, {z0.s - z3.s}, z0.s
sudot za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b
sudot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b
usdot za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b
usdot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b
udot za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b
udot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b
sdot za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b
sdot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b
usmlall za.s[w8, 0:3], z0.b, z0.b
usmlall za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b
sumopa za0.s, p0/m, p0/m, z0.b, z0.b
sumops za0.s, p0/m, p0/m, z0.b, z0.b
usmopa za0.s, p0/m, p0/m, z0.b, z0.b
usmops za0.s, p0/m, p0/m, z0.b, z0.b
sunpk {z0.h - z1.h}, z0.b
sunpk {z0.h - z3.h}, {z0.b, z1.b}
uunpk {z0.h - z1.h}, z0.b
uunpk {z0.h - z3.h}, {z0.b, z1.b}
st1b {z0.b, z1.b}, pn8, [x0]
st1b {z0.b - z3.b}, pn8, [x0]
stnt1b {z0.b, z1.b}, pn8, [x0]
stnt1b {z0.b - z3.b}, pn8, [x0]
suvdot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]
uvdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]
uvdot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]
svdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]
svdot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]
usvdot za.s[w8, 0, vgx4], {z0.b - z3.b}, z0.b[0]
uzp  {z0.q - z1.q}, z0.q, z0.q
uzp {z0.s - z3.s}, {z0.s - z3.s}
uzp1 z31.s, z31.s, z31.s
uzp2 z31.s, z31.s, z31.s
zero za.d[w8, 0, vgx2]
zero {zt0}
zip1 z0.s, z0.s, z0.s
zip2 z0.s, z0.s, z0.s
zip {z0.q - z1.q}, z0.q, z0.q
zip {z0.q - z3.q}, {z0.q - z3.q}
rprfm #0, x0, [x0]

// SME load instructions.
ld1q {za0h.q[w12, 0]}, p0/z, [x0, x0, lsl #4]
ld1w {za0v.s[w12, 0]}, p0/z, [x0]
ld1w {z0.s - z3.s}, pn8/z, [x0, x0, lsl #2]
ld1w {z0.s, z4.s, z8.s, z12.s}, pn8/z, [x0, x0, lsl #2]
ldnt1w {z0.s, z4.s, z8.s, z12.s}, pn8/z, [x0]
ldr zt0, [x0]
ldr za[w12, #0], [x0]

// SME store instructions.
st1q {za0h.q[w12, 0]}, p0, [x0, x0, lsl #4]
st1w {za0h.s[w12, 0]}, p0, [x0]
st1w {z0.s - z3.s}, pn8, [x0, x0, lsl #2]
st1w {z0.s, z4.s, z8.s, z12.s}, pn8, [x0]
st1w {z0.s, z4.s, z8.s, z12.s}, pn8, [x0, x0, lsl #2]
stnt1w {z0.s - z3.s}, pn8, [x0, x0, lsl #2]
stnt1w {z0.s, z4.s, z8.s, z12.s}, pn8, [x0]
str zt0, [x0]
str za[w12, #0], [x0]

// SYS instructions.
cpp rctx, x2
