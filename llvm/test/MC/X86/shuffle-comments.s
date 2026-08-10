# RUN: llvm-mc %s -triple=x86_64-unknown-unknown | FileCheck %s

palignr $8, %xmm0, %xmm1
# CHECK: xmm1 = xmm0[8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7]
palignr $8, (%rax), %xmm1
# CHECK: xmm1 = mem[8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7]

palignr $16, %xmm0, %xmm1
# CHECK: xmm1 = xmm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
palignr $16, (%rax), %xmm1
# CHECK: xmm1 = xmm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

palignr $0, %xmm0, %xmm1
# CHECK: xmm1 = xmm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
palignr $0, (%rax), %xmm1
# CHECK: xmm1 = mem[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

vpalignr $8, %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm0[8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7]
vpalignr $8, (%rax), %xmm1, %xmm2
# CHECK: xmm2 = mem[8,9,10,11,12,13,14,15],xmm1[0,1,2,3,4,5,6,7]

vpalignr $16, %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
vpalignr $16, (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

vpalignr $0, %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
vpalignr $0, (%rax), %xmm1, %xmm2
# CHECK: xmm2 = mem[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

vpalignr $8, %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm0[8,9,10,11,12,13,14,15],ymm1[0,1,2,3,4,5,6,7],ymm0[24,25,26,27,28,29,30,31],ymm1[16,17,18,19,20,21,22,23]
vpalignr $8, (%rax), %ymm1, %ymm2
# CHECK: ymm2 = mem[8,9,10,11,12,13,14,15],ymm1[0,1,2,3,4,5,6,7],mem[24,25,26,27,28,29,30,31],ymm1[16,17,18,19,20,21,22,23]

vpalignr $16, %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
vpalignr $16, (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

vpalignr $0, %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
vpalignr $0, (%rax), %ymm1, %ymm2
# CHECK: ymm2 = mem[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

pshufd $27, %xmm0, %xmm1
# CHECK: xmm1 = xmm0[3,2,1,0]
pshufd $27, (%rax), %xmm1
# CHECK: xmm1 = mem[3,2,1,0]

vpshufd $27, %xmm0, %xmm1
# CHECK: xmm1 = xmm0[3,2,1,0]
vpshufd $27, (%rax), %xmm1
# CHECK: xmm1 = mem[3,2,1,0]

vpshufd $27, %ymm0, %ymm1
# CHECK: ymm1 = ymm0[3,2,1,0,7,6,5,4]
vpshufd $27, (%rax), %ymm1
# CHECK: ymm1 = mem[3,2,1,0,7,6,5,4]

punpcklbw %xmm0, %xmm1
# CHECK: xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
punpcklbw (%rax), %xmm1
# CHECK: xmm1 = xmm1[0],mem[0],xmm1[1],mem[1],xmm1[2],mem[2],xmm1[3],mem[3],xmm1[4],mem[4],xmm1[5],mem[5],xmm1[6],mem[6],xmm1[7],mem[7]

vpunpcklbw %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
vpunpcklbw (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],mem[0],xmm1[1],mem[1],xmm1[2],mem[2],xmm1[3],mem[3],xmm1[4],mem[4],xmm1[5],mem[5],xmm1[6],mem[6],xmm1[7],mem[7]

vpunpcklbw %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],ymm0[0],ymm1[1],ymm0[1],ymm1[2],ymm0[2],ymm1[3],ymm0[3],ymm1[4],ymm0[4],ymm1[5],ymm0[5],ymm1[6],ymm0[6],ymm1[7],ymm0[7],ymm1[16],ymm0[16],ymm1[17],ymm0[17],ymm1[18],ymm0[18],ymm1[19],ymm0[19],ymm1[20],ymm0[20],ymm1[21],ymm0[21],ymm1[22],ymm0[22],ymm1[23],ymm0[23]
vpunpcklbw (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],mem[0],ymm1[1],mem[1],ymm1[2],mem[2],ymm1[3],mem[3],ymm1[4],mem[4],ymm1[5],mem[5],ymm1[6],mem[6],ymm1[7],mem[7],ymm1[16],mem[16],ymm1[17],mem[17],ymm1[18],mem[18],ymm1[19],mem[19],ymm1[20],mem[20],ymm1[21],mem[21],ymm1[22],mem[22],ymm1[23],mem[23]

punpckhbw %xmm0, %xmm1
# CHECK: xmm1 = xmm1[8],xmm0[8],xmm1[9],xmm0[9],xmm1[10],xmm0[10],xmm1[11],xmm0[11],xmm1[12],xmm0[12],xmm1[13],xmm0[13],xmm1[14],xmm0[14],xmm1[15],xmm0[15]
punpckhbw (%rax), %xmm1
# CHECK: xmm1 = xmm1[8],mem[8],xmm1[9],mem[9],xmm1[10],mem[10],xmm1[11],mem[11],xmm1[12],mem[12],xmm1[13],mem[13],xmm1[14],mem[14],xmm1[15],mem[15]

vpunpckhbw %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[8],xmm0[8],xmm1[9],xmm0[9],xmm1[10],xmm0[10],xmm1[11],xmm0[11],xmm1[12],xmm0[12],xmm1[13],xmm0[13],xmm1[14],xmm0[14],xmm1[15],xmm0[15]
vpunpckhbw (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[8],mem[8],xmm1[9],mem[9],xmm1[10],mem[10],xmm1[11],mem[11],xmm1[12],mem[12],xmm1[13],mem[13],xmm1[14],mem[14],xmm1[15],mem[15]

vpunpckhbw %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[8],ymm0[8],ymm1[9],ymm0[9],ymm1[10],ymm0[10],ymm1[11],ymm0[11],ymm1[12],ymm0[12],ymm1[13],ymm0[13],ymm1[14],ymm0[14],ymm1[15],ymm0[15],ymm1[24],ymm0[24],ymm1[25],ymm0[25],ymm1[26],ymm0[26],ymm1[27],ymm0[27],ymm1[28],ymm0[28],ymm1[29],ymm0[29],ymm1[30],ymm0[30],ymm1[31],ymm0[31]
vpunpckhbw (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[8],mem[8],ymm1[9],mem[9],ymm1[10],mem[10],ymm1[11],mem[11],ymm1[12],mem[12],ymm1[13],mem[13],ymm1[14],mem[14],ymm1[15],mem[15],ymm1[24],mem[24],ymm1[25],mem[25],ymm1[26],mem[26],ymm1[27],mem[27],ymm1[28],mem[28],ymm1[29],mem[29],ymm1[30],mem[30],ymm1[31],mem[31]

punpcklwd %xmm0, %xmm1
# CHECK: xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
punpcklwd (%rax), %xmm1
# CHECK: xmm1 = xmm1[0],mem[0],xmm1[1],mem[1],xmm1[2],mem[2],xmm1[3],mem[3]

vpunpcklwd %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
vpunpcklwd (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],mem[0],xmm1[1],mem[1],xmm1[2],mem[2],xmm1[3],mem[3]

vpunpcklwd %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],ymm0[0],ymm1[1],ymm0[1],ymm1[2],ymm0[2],ymm1[3],ymm0[3],ymm1[8],ymm0[8],ymm1[9],ymm0[9],ymm1[10],ymm0[10],ymm1[11],ymm0[11]
vpunpcklwd (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],mem[0],ymm1[1],mem[1],ymm1[2],mem[2],ymm1[3],mem[3],ymm1[8],mem[8],ymm1[9],mem[9],ymm1[10],mem[10],ymm1[11],mem[11]

punpckhwd %xmm0, %xmm1
# CHECK: xmm1 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
punpckhwd (%rax), %xmm1
# CHECK: xmm1 = xmm1[4],mem[4],xmm1[5],mem[5],xmm1[6],mem[6],xmm1[7],mem[7]

vpunpckhwd %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
vpunpckhwd (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[4],mem[4],xmm1[5],mem[5],xmm1[6],mem[6],xmm1[7],mem[7]

vpunpckhwd %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[4],ymm0[4],ymm1[5],ymm0[5],ymm1[6],ymm0[6],ymm1[7],ymm0[7],ymm1[12],ymm0[12],ymm1[13],ymm0[13],ymm1[14],ymm0[14],ymm1[15],ymm0[15]
vpunpckhwd (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[4],mem[4],ymm1[5],mem[5],ymm1[6],mem[6],ymm1[7],mem[7],ymm1[12],mem[12],ymm1[13],mem[13],ymm1[14],mem[14],ymm1[15],mem[15]

punpckldq %xmm0, %xmm1
# CHECK: xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
punpckldq (%rax), %xmm1
# CHECK: xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]

vpunpckldq %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
vpunpckldq (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],mem[0],xmm1[1],mem[1]

vpunpckldq %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],ymm0[0],ymm1[1],ymm0[1],ymm1[4],ymm0[4],ymm1[5],ymm0[5]
vpunpckldq (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],mem[0],ymm1[1],mem[1],ymm1[4],mem[4],ymm1[5],mem[5]

punpckhdq %xmm0, %xmm1
# CHECK: xmm1 = xmm1[2],xmm0[2],xmm1[3],xmm0[3]
punpckhdq (%rax), %xmm1
# CHECK: xmm1 = xmm1[2],mem[2],xmm1[3],mem[3]

vpunpckhdq %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[2],xmm0[2],xmm1[3],xmm0[3]
vpunpckhdq (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[2],mem[2],xmm1[3],mem[3]

vpunpckhdq %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[2],ymm0[2],ymm1[3],ymm0[3],ymm1[6],ymm0[6],ymm1[7],ymm0[7]
vpunpckhdq (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[2],mem[2],ymm1[3],mem[3],ymm1[6],mem[6],ymm1[7],mem[7]

punpcklqdq %xmm0, %xmm1
# CHECK: xmm1 = xmm1[0],xmm0[0]
punpcklqdq (%rax), %xmm1
# CHECK: xmm1 = xmm1[0],mem[0]

vpunpcklqdq %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],xmm0[0]
vpunpcklqdq (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],mem[0]

vpunpcklqdq %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],ymm0[0],ymm1[2],ymm0[2]
vpunpcklqdq (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],mem[0],ymm1[2],mem[2]

punpckhqdq %xmm0, %xmm1
# CHECK: xmm1 = xmm1[1],xmm0[1]
punpckhqdq (%rax), %xmm1
# CHECK: xmm1 = xmm1[1],mem[1]

vpunpckhqdq %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[1],xmm0[1]
vpunpckhqdq (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[1],mem[1]

vpunpckhqdq %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[1],ymm0[1],ymm1[3],ymm0[3]
vpunpckhqdq (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[1],mem[1],ymm1[3],mem[3]

unpcklps %xmm0, %xmm1
# CHECK: xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
unpcklps (%rax), %xmm1
# CHECK: xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]

vunpcklps %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
vunpcklps (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],mem[0],xmm1[1],mem[1]

vunpcklps %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],ymm0[0],ymm1[1],ymm0[1],ymm1[4],ymm0[4],ymm1[5],ymm0[5]
vunpcklps (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],mem[0],ymm1[1],mem[1],ymm1[4],mem[4],ymm1[5],mem[5]

unpckhps %xmm0, %xmm1
# CHECK: xmm1 = xmm1[2],xmm0[2],xmm1[3],xmm0[3]
unpckhps (%rax), %xmm1
# CHECK: xmm1 = xmm1[2],mem[2],xmm1[3],mem[3]

vunpckhps %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[2],xmm0[2],xmm1[3],xmm0[3]
vunpckhps (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[2],mem[2],xmm1[3],mem[3]

vunpckhps %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[2],ymm0[2],ymm1[3],ymm0[3],ymm1[6],ymm0[6],ymm1[7],ymm0[7]
vunpckhps (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[2],mem[2],ymm1[3],mem[3],ymm1[6],mem[6],ymm1[7],mem[7]

unpcklpd %xmm0, %xmm1
# CHECK: xmm1 = xmm1[0],xmm0[0]
unpcklpd (%rax), %xmm1
# CHECK: xmm1 = xmm1[0],mem[0]

vunpcklpd %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],xmm0[0]
vunpcklpd (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],mem[0]

vunpcklpd %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],ymm0[0],ymm1[2],ymm0[2]
vunpcklpd (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[0],mem[0],ymm1[2],mem[2]

unpckhpd %xmm0, %xmm1
# CHECK: xmm1 = xmm1[1],xmm0[1]
unpckhpd (%rax), %xmm1
# CHECK: xmm1 = xmm1[1],mem[1]

vunpckhpd %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[1],xmm0[1]
vunpckhpd (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[1],mem[1]

vunpckhpd %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[1],ymm0[1],ymm1[3],ymm0[3]
vunpckhpd (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[1],mem[1],ymm1[3],mem[3]

shufps $27, %xmm0, %xmm1
# CHECK: xmm1 = xmm1[3,2],xmm0[1,0]
shufps $27, (%rax), %xmm1
# CHECK: xmm1 = xmm1[3,2],mem[1,0]

vshufps $27, %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[3,2],xmm0[1,0]
vshufps $27, (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[3,2],mem[1,0]

vshufps $27, %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[3,2],ymm0[1,0],ymm1[7,6],ymm0[5,4]
vshufps $27, (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[3,2],mem[1,0],ymm1[7,6],mem[5,4]

shufpd $3, %xmm0, %xmm1
# CHECK: xmm1 = xmm1[1],xmm0[1]
shufpd $3, (%rax), %xmm1
# CHECK: xmm1 = xmm1[1],mem[1]

vshufpd $3, %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[1],xmm0[1]
vshufpd $3, (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[1],mem[1]

vshufpd $11, %ymm0, %ymm1, %ymm2
# CHECK: ymm2 = ymm1[1],ymm0[1],ymm1[2],ymm0[3]
vshufpd $11, (%rax), %ymm1, %ymm2
# CHECK: ymm2 = ymm1[1],mem[1],ymm1[2],mem[3]

vinsertps $16, %xmm0, %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],xmm0[0],xmm1[2,3]
vinsertps $16, (%rax), %xmm1, %xmm2
# CHECK: xmm2 = xmm1[0],mem[0],xmm1[2,3]
