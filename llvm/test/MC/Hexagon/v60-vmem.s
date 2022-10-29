#RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -filetype=obj -mhvx %s | \
#RUN: llvm-objdump --no-print-imm-hex --triple=hexagon --mcpu=hexagonv60 --mattr=+hvx -d - | \
#RUN: FileCheck %s

#CHECK: 292cc11b { vmem(r12++#1) = v27 }
{
  vmem(r12++#1)=v27
}

#CHECK: 294dc319 { v25 = vmem(r13++#3):nt }
{
  v25=vmem(r13++#3):nt
}

#CHECK: 2904c1fb { v27 = vmemu(r4++#1) }
{
  v27=vmemu(r4++#1)
}

#CHECK: 291dc01f { v31 = vmem(r29++#0) }
{
  v31=vmem(r29++#0)
}

#CHECK: 293ec0ff { vmemu(r30++#0) = v31 }
{
  vmemu(r30++#0)=v31
}

#CHECK: 296ec411 { vmem(r14++#-4):nt = v17 }
{
  vmem(r14++#-4):nt=v17
}

#CHECK: 29fec62f { if (!p0) vmem(r30++#-2):nt = v15 }
{
  if (!p0) vmem(r30++#-2):nt=v15
}

#CHECK: 29f9c914 { if (p1) vmem(r25++#1):nt = v20 }
{
  if (p1) vmem(r25++#1):nt=v20
}

#CHECK: 2984de30 { if (!q3) vmem(r4++#-2) = v16 }
{
  if (!q3) vmem(r4++#-2)=v16
}

#CHECK: 2992dd1f { if (q3) vmem(r18++#-3) = v31 }
{
  if (q3) vmem(r18++#-3)=v31
}

#CHECK: 29c9c425 { if (!q0) vmem(r9++#-4):nt = v5 }
{
  if (!q0) vmem(r9++#-4):nt=v5
}

#CHECK: 29d1cf11 { if (q1) vmem(r17++#-1):nt = v17 }
{
  if (q1) vmem(r17++#-1):nt=v17
}

#CHECK: 29a7c328 { if (!p0) vmem(r7++#3) = v8 }
{
  if (!p0) vmem(r7++#3)=v8
}

#CHECK: 29b6cc1d { if (p1) vmem(r22++#-4) = v29 }
{
  if (p1) vmem(r22++#-4)=v29
}

#CHECK: 29abc5fe { if (!p0) vmemu(r11++#-3) = v30 }
{
  if (!p0) vmemu(r11++#-3)=v30
}

#CHECK: 29b8d5c4 { if (p2) vmemu(r24++#-3) = v4 }
{
  if (p2) vmemu(r24++#-3)=v4
}

#CHECK: 2860e407 { vmem(r0+#-4):nt = v7 }
{
  vmem(r0+#-4):nt=v7
}

#CHECK: 2830e2e7 { vmemu(r16+#-6) = v7 }
{
  vmemu(r16+#-6)=v7
}

#CHECK: 2839c316 { vmem(r25+#3) = v22 }
{
  vmem(r25+#3)=v22
}
#CHECK: 284be316 { v22 = vmem(r11+#-5):nt }
{
  v22=vmem(r11+#-5):nt
}

#CHECK: 280ec1e6 { v6 = vmemu(r14+#1) }
{
  v6=vmemu(r14+#1)
}

#CHECK: 280ae50c { v12 = vmem(r10+#-3) }
{
  v12=vmem(r10+#-3)
}

#CHECK: 2b62e005 { vmem(r2++m1):nt = v5 }
{
  vmem(r2++m1):nt=v5
}

#CHECK: 2b28e0f2 { vmemu(r8++m1) = v18 }
{
  vmemu(r8++m1)=v18
}

#CHECK: 2b42e019 { v25 = vmem(r2++m1):nt }
{
  v25=vmem(r2++m1):nt
}

#CHECK: 2b2ce009 { vmem(r12++m1) = v9 }
{
  vmem(r12++m1)=v9
}

#CHECK: 2b03c005 { v5 = vmem(r3++m0) }
{
  v5=vmem(r3++m0)
}


#CHECK: 2b0ec0f5 { v21 = vmemu(r14++m0) }
{
  v21=vmemu(r14++m0)
}

#CHECK: 2be8c022 { if (!p0) vmem(r8++m0):nt = v2 }
{
  if (!p0) vmem(r8++m0):nt=v2
}

#CHECK: 2bebd813 { if (p3) vmem(r11++m0):nt = v19 }
{
  if (p3) vmem(r11++m0):nt=v19
}

#CHECK: 2ba5e0e7 { if (!p0) vmemu(r5++m1) = v7 }
{
  if (!p0) vmemu(r5++m1)=v7
}

#CHECK: 2ba4f0dd { if (p2) vmemu(r4++m1) = v29 }
{
  if (p2) vmemu(r4++m1)=v29
}

#CHECK: 2ba4e828 { if (!p1) vmem(r4++m1) = v8 }
{
  if (!p1) vmem(r4++m1)=v8
}

#CHECK: 2bbae803 { if (p1) vmem(r26++m1) = v3 }
{
  if (p1) vmem(r26++m1)=v3
}

#CHECK: 2bc9c027 { if (!q0) vmem(r9++m0):nt = v7 }
{
  if (!q0) vmem(r9++m0):nt=v7
}

#CHECK: 2bcfc001 { if (q0) vmem(r15++m0):nt = v1 }
{
  if (q0) vmem(r15++m0):nt=v1
}

#CHECK: 2b97f031 { if (!q2) vmem(r23++m1) = v17 }
{
  if (!q2) vmem(r23++m1)=v17
}

#CHECK: 2b8ad809 { if (q3) vmem(r10++m0) = v9 }
{
  if (q3) vmem(r10++m0)=v9
}

#CHECK: 28c7f438 { if (!q2) vmem(r7+#-4):nt = v24 }
{
  if (!q2) vmem(r7+#-4):nt=v24
}

#CHECK: 28d1eb15 { if (q1) vmem(r17+#-5):nt = v21 }
{
  if (q1) vmem(r17+#-5):nt=v21
}

#CHECK: 289cfe2b { if (!q3) vmem(r28+#-2) = v11 }
{
  if (!q3) vmem(r28+#-2)=v11
}

#CHECK: 288eef0f { if (q1) vmem(r14+#-1) = v15 }
{
  if (q1) vmem(r14+#-1)=v15
}

#CHECK: 28a2d1e1 { if (!p2) vmemu(r2+#1) = v1 }
{
  if (!p2) vmemu(r2+#1)=v1
}

#CHECK: 28bcf4db { if (p2) vmemu(r28+#-4) = v27 }
{
  if (p2) vmemu(r28+#-4)=v27
}

#CHECK: 28b2c925 { if (!p1) vmem(r18+#1) = v5 }
{
  if (!p1) vmem(r18+#1)=v5
}

#CHECK: 28afe41a { if (p0) vmem(r15+#-4) = v26 }
{
  if (p0) vmem(r15+#-4)=v26
}

#CHECK: 28f7fd3a { if (!p3) vmem(r23+#-3):nt = v26 }
{
  if (!p3) vmem(r23+#-3):nt=v26
}

#CHECK: 28f5fd10 { if (p3) vmem(r21+#-3):nt = v16 }
{
  if (p3) vmem(r21+#-3):nt=v16
}

#CHECK: 2945c440 v0.tmp = vmem(r5++#-4):nt }
{
  v0.tmp=vmem(r5++#-4):nt
  v26=v0
}

#CHECK: 2942c338 v24.cur = vmem(r2++#3):nt }
{
  v24.cur=vmem(r2++#3):nt
  v6=v24
}

#CHECK: 2908c157 v23.tmp = vmem(r8++#1) }
{
  v25=v23
  v23.tmp=vmem(r8++#1)
}

#CHECK: 2903c72d v13.cur = vmem(r3++#-1) }
{
  v13.cur=vmem(r3++#-1)
  v21=v13
}

#CHECK: 2855c743 v3.tmp = vmem(r21+#7):nt }
{
  v3.tmp=vmem(r21+#7):nt
  v21=v3
}

#CHECK: 2856e025 v5.cur = vmem(r22+#-8):nt }
{
  v5.cur=vmem(r22+#-8):nt
  v29=v5
}

#CHECK: 2802c555 v21.tmp = vmem(r2+#5) }
{
  v31=v21
  v21.tmp=vmem(r2+#5)
}

#CHECK: 2814e12a v10.cur = vmem(r20+#-7) }
{
  v9=v10
  v10.cur=vmem(r20+#-7)
}


#CHECK: 2b52c02c v12.cur = vmem(r18++m0):nt }
{
  v12.cur=vmem(r18++m0):nt
  v25=v12
}

#CHECK: 2b4ae043 v3.tmp = vmem(r10++m1):nt }
{
  v25=v3
  v3.tmp=vmem(r10++m1):nt
}

#CHECK: 2b06c025 v5.cur = vmem(r6++m0) }
{
  v5.cur=vmem(r6++m0)
  v10=v5
}

#CHECK: 2b17e048 v8.tmp = vmem(r23++m1) }
{
  v8.tmp=vmem(r23++m1)
  v28=v8
}

#CHECK: 282ee422 vmem(r14+#-4) = v14.new }
{
  v14 = v14
  vmem(r14+#-4)=v14.new
}

#CHECK: 2866e222 vmem(r6+#-6):nt = v16.new }
{
  v16 = v8
  vmem(r6+#-6):nt=v16.new
}

#CHECK: 28b1cd42 if (p1) vmem(r17+#5) = v17.new }
{
  v17 = v25
  if(p1)vmem(r17+#5)=v17.new
}

#CHECK: 28bbeb6a if (!p1) vmem(r27+#-5) = v17.new }
{
  v17 = v15
  if(!p1)vmem(r27+#-5)=v17.new
}

#CHECK: 28e4d252 if (p2) vmem(r4+#2):nt = v24.new }
{
  v24 = v10
  if(p2)vmem(r4+#2):nt=v24.new
}

#CHECK: 28f8d17a if (!p2) vmem(r24+#1):nt = v4.new }
{
  v4 = v8
  if(!p2)vmem(r24+#1):nt=v4.new
}

#CHECK: 2924c322 vmem(r4++#3) = v4.new }
{
  v4 = v3
  vmem(r4++#3)=v4.new
}

#CHECK: 2961c122 vmem(r1++#1):nt = v7.new }
{
  v7 = v8
  vmem(r1++#1):nt=v7.new
}

#CHECK: 29a6d042 if (p2) vmem(r6++#0) = v11.new }
{
  v11 = v13
  if(p2)vmem(r6++#0)=v11.new
}

#CHECK: 29a2cb6a if (!p1) vmem(r2++#3) = v25.new }
{
  v25 = v17
  if(!p1)vmem(r2++#3)=v25.new
}

#CHECK: 29f5c952 if (p1) vmem(r21++#1):nt = v14.new }
{
  v14 = v13
  if(p1)vmem(r21++#1):nt=v14.new
}

#CHECK: 29f7cd7a if (!p1) vmem(r23++#-3):nt = v1.new }
{
  v1 = v0
  if(!p1)vmem(r23++#-3):nt=v1.new
}

#CHECK: 2b3ec022 vmem(r30++m0) = v10.new }
{
  v10 = v23
  vmem(r30++m0)=v10.new
}

#CHECK: 2b6fc022 vmem(r15++m0):nt = v19.new }
{
  v19 = v20
  vmem(r15++m0):nt=v19.new
}

#CHECK: 2bb7f042 if (p2) vmem(r23++m1) = v6.new }
{
  v6 = v30
  if(p2)vmem(r23++m1)=v6.new
}

#CHECK: 2ba2f06a if (!p2) vmem(r2++m1) = v12.new }
{
  v12 = v9
  if(!p2)vmem(r2++m1)=v12.new
}

#CHECK: 2be7e852 if (p1) vmem(r7++m1):nt = v3.new }
{
  v3 = v13
  if(p1)vmem(r7++m1):nt=v3.new
}

#CHECK: 2bfdd07a if (!p2) vmem(r29++m0):nt = v29.new }
{
  v29 = v9
  if(!p2)vmem(r29++m0):nt=v29.new
}
