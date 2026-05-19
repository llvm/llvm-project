addva za0.s, p0/m, p0/m, z0.s
bmopa za0.s, p0/m, p0/m, z0.s, z0.s
fmopa za0.s, p0/m, p0/m, z0.h, z0.h
sub za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s
fclamp {z0.h - z1.h}, z0.h, z1.h
scvtf {z0.s, z1.s}, {z0.s, z1.s}
sqdmulh {z0.h, z1.h}, {z0.h, z1.h}, z0.h
sqcvt z0.b, {z0.s - z3.s}
sudot za.s[w8, 0, vgx2], {z0.b, z1.b}, z0.b
svdot za.s[w8, 0, vgx2], {z0.h, z1.h}, z0.h[0]
smlall za.s[w8, 0:3, vgx2], {z0.b, z1.b}, z0.b
mova {z0.s, z1.s}, za0h.s[w12, 0:1]
mova za0h.s[w12, 0:1], {z0.s, z1.s}
movt x0, zt0[0]
movt zt0[0], x0
luti2 {z0.h - z1.h}, zt0, z0[0]
zero za.d[w8, 0, vgx2]
zero {zt0}
ldr za[w12, #0], [x0]
str za[w12, #0], [x0]
rprfm #0, x0, [x0]
