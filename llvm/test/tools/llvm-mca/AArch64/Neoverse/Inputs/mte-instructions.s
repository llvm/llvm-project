irg	x0, x1
irg	sp, x1
irg	x0, sp
irg	x0, x1, x2
irg	sp, x1, x2
addg	x0, x1, #0, #1
addg	sp, x2, #32, #3
addg	x0, sp, #64, #5
addg	x3, x4, #1008, #6
addg	x5, x6, #112, #15
subg	x0, x1, #0, #1
subg	sp, x2, #32, #3
subg	x0, sp, #64, #5
subg	x3, x4, #1008, #6
subg	x5, x6, #112, #15
gmi	x0, x1, x2
gmi	x3, sp, x4
gmi	xzr, x0, x30
gmi	x30, x0, xzr
subp	x0, x1, x2
subps	x0, x1, x2
subp	x0, sp, sp
subps	x0, sp, sp
subps	xzr, x0, x1
subps	xzr, sp, sp
stg	x0, [x1, #-4096]
stg	x1, [x2, #4080]
stg	x2, [sp, #16]
stg	x3, [x1]
stg	sp, [x1]
stzg	x0, [x1, #-4096]
stzg	x1, [x2, #4080]
stzg	x2, [sp, #16]
stzg	x3, [x1]
stzg	sp, [x1]
stg	x0, [x1, #-4096]!
stg	x1, [x2, #4080]!
stg	x2, [sp, #16]!
stg	sp, [sp, #16]!
stzg	x0, [x1, #-4096]!
stzg	x1, [x2, #4080]!
stzg	x2, [sp, #16]!
stzg	sp, [sp, #16]!
stg	x0, [x1], #-4096
stg	x1, [x2], #4080
stg	x2, [sp], #16
stg	sp, [sp], #16
stzg	x0, [x1], #-4096
stzg	x1, [x2], #4080
stzg	x2, [sp], #16
stzg	sp, [sp], #16
st2g	x0, [x1, #-4096]
st2g	x1, [x2, #4080]
st2g	x2, [sp, #16]
st2g	x3, [x1]
st2g	sp, [x1]
stz2g	x0, [x1, #-4096]
stz2g	x1, [x2, #4080]
stz2g	x2, [sp, #16]
stz2g	x3, [x1]
stz2g	sp, [x1]
st2g	x0, [x1, #-4096]!
st2g	x1, [x2, #4080]!
st2g	x2, [sp, #16]!
st2g	sp, [sp, #16]!
stz2g	x0, [x1, #-4096]!
stz2g	x1, [x2, #4080]!
stz2g	x2, [sp, #16]!
stz2g	sp, [sp, #16]!
st2g	x0, [x1], #-4096
st2g	x1, [x2], #4080
st2g	x2, [sp], #16
st2g	sp, [sp], #16
stz2g	x0, [x1], #-4096
stz2g	x1, [x2], #4080
stz2g	x2, [sp], #16
stz2g	sp, [sp], #16
stgp	x0, x1, [x2, #-1024]
stgp	x0, x1, [x2, #1008]
stgp	x0, x1, [sp, #16]
stgp	xzr, x1, [x2, #16]
stgp	x0, xzr, [x2, #16]
stgp	x0, xzr, [x2]
stgp	x0, x1, [x2, #-1024]!
stgp	x0, x1, [x2, #1008]!
stgp	x0, x1, [sp, #16]!
stgp	xzr, x1, [x2, #16]!
stgp	x0, xzr, [x2, #16]!
stgp	x0, x1, [x2], #-1024
stgp	x0, x1, [x2], #1008
stgp	x0, x1, [sp], #16
stgp	xzr, x1, [x2], #16
stgp	x0, xzr, [x2], #16
ldg	x0, [x1]
ldg	x2, [sp, #-4096]
ldg	x3, [x4, #4080]
ldgm	x0, [x1]
ldgm	x1, [sp]
ldgm	xzr, [x2]
stgm	x0, [x1]
stgm	x1, [sp]
stgm	xzr, [x2]
stzgm	x0, [x1]
stzgm	x1, [sp]
stzgm	xzr, [x2]
