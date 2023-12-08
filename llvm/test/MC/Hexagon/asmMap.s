#RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

# Make sure that the assembler mapped instructions are being handled correctly.

#CHECK: 3c56c000 { memw(r22+#0) = #0
memw(r22)=#0

#CHECK: 3c23e05f { memh(r3+#0) = #-33
memh(r3)=#-33

#CHECK: 3c07c012 { memb(r7+#0) = #18
memb(r7)=#18

#CHECK: 4101c008 { if (p0) r8 = memb(r1+#0)
if (p0) r8=memb(r1)

#CHECK: 4519d817 { if (!p3) r23 = memb(r25+#0)
if (!p3) r23=memb(r25)

#CHECK: 412dc002 { if (p0) r2 = memub(r13+#0)
if (p0) r2=memub(r13)

#CHECK: 453cc01a { if (!p0) r26 = memub(r28+#0)
if (!p0) r26=memub(r28)

#CHECK: 416bc818 { if (p1) r24 = memuh(r11+#0)
if (p1) r24=memuh(r11)

#CHECK: 457fc012 { if (!p0) r18 = memuh(r31+#0)
if (!p0) r18=memuh(r31)

#CHECK: 455dc014 { if (!p0) r20 = memh(r29+#0)
if (!p0) r20=memh(r29)

#CHECK: 415dc01d { if (p0) r29 = memh(r29+#0)
if (p0) r29=memh(r29)

#CHECK: 4583c01d { if (!p0) r29 = memw(r3+#0)
if (!p0) r29=memw(r3)

#CHECK: 419bd01e { if (p2) r30 = memw(r27+#0)
if (p2) r30=memw(r27)

#CHECK: 90e2c018 { r25:24 = membh(r2+#0)
r25:24=membh(r2)

#CHECK: 902bc006 { r6 = membh(r11+#0)
r6=membh(r11)

#CHECK: 90a2c01c { r29:28 = memubh(r2+#0)
r29:28=memubh(r2)

#CHECK: 906ec00d { r13 = memubh(r14+#0)
r13=memubh(r14)

#CHECK: 91dac00c { r13:12 = memd(r26+#0)
r13:12=memd(r26)

#CHECK: 919bc004 { r4 = memw(r27+#0)
r4=memw(r27)

#CHECK: 914cc005 { r5 = memh(r12+#0)
r5=memh(r12)

#CHECK: 9176c010 { r16 = memuh(r22+#0)
r16=memuh(r22)

#CHECK: 910bc017 { r23 = memb(r11+#0)
r23=memb(r11)

#CHECK: 912bc01b { r27 = memub(r11+#0)
r27=memub(r11)

#CHECK: 404ede01 { if (p1) memh(r14+#0) = r30
if (p1) memh(r14)=r30

#CHECK: 4449d900 { if (!p0) memh(r9+#0) = r25
if (!p0) memh(r9)=r25

#CHECK: 400ecd00 { if (p0) memb(r14+#0) = r13
if (p0) memb(r14)=r13

#CHECK: 440bcc01 { if (!p1) memb(r11+#0) = r12
if (!p1) memb(r11)=r12

#CHECK: 41d0d804 { if (p3) r5:4 = memd(r16+#0)
if (p3) r5:4=memd(r16)

#CHECK: 45d9c00c { if (!p0) r13:12 = memd(r25+#0)
if (!p0) r13:12=memd(r25)

#CHECK: 385ee06d { if (p3) memw(r30+#0) = #-19
if (p3) memw(r30)=#-19

#CHECK: 38c6c053 { if (!p2) memw(r6+#0) = #19
if (!p2) memw(r6)=#19

#CHECK: 381fc034 { if (p1) memb(r31+#0) = #20
if (p1) memb(r31)=#20

#CHECK: 389dc010 { if (!p0) memb(r29+#0) = #16
if (!p0) memb(r29)=#16

#CHECK: 3833e019 { if (p0) memh(r19+#0) = #-7
if (p0) memh(r19)=#-7

#CHECK: 38b7c013 { if (!p0) memh(r23+#0) = #19
if (!p0) memh(r23)=#19

#CHECK: 4488d401 { if (!p1) memw(r8+#0) = r20
if (!p1) memw(r8)=r20

#CHECK: 409ddc02 { if (p2) memw(r29+#0) = r28
if (p2) memw(r29)=r28

#CHECK: 446fc301 { if (!p1) memh(r15+#0) = r3.h
if (!p1) memh(r15)=r3.h

#CHECK: 406dc201 { if (p1) memh(r13+#0) = r2.h
if (p1) memh(r13)=r2.h

#CHECK: 40d9c601 { if (p1) memd(r25+#0) = r7:6
if (p1) memd(r25)=r7:6

#CHECK: 44dad803 { if (!p3) memd(r26+#0) = r25:24
if (!p3) memd(r26)=r25:24

#CHECK: 3e21c011 { memh(r1+#0) += r17
memh(r1)+=r17

#CHECK: 3e4fc019 { memw(r15+#0) += r25
memw(r15)+=r25

#CHECK: 3e5dc022 { memw(r29+#0) -= r2
memw(r29)-=r2

#CHECK: 3e04c004 { memb(r4+#0) += r4
memb(r4)+=r4

#CHECK: 3f53c016 { memw(r19+#0) += #22
memw(r19)+=#22

#CHECK: 3f24c01e { memh(r4+#0) += #30
memh(r4)+=#30

#CHECK: 3e27c02d { memh(r7+#0) -= r13
memh(r7)-=r13

#CHECK: 3e1ec032 { memb(r30+#0) -= r18
memb(r30)-=r18

#CHECK: 3e49c05b { memw(r9+#0) &= r27
memw(r9)&=r27

#CHECK: 3e2dc040 { memh(r13+#0) &= r0
memh(r13)&=r0

#CHECK: 3e05c046 { memb(r5+#0) &= r6
memb(r5)&=r6

#CHECK: 3e45c06a { memw(r5+#0) |= r10
memw(r5)|=r10

#CHECK: 3e21c07e { memh(r1+#0) |= r30
memh(r1)|=r30

#CHECK: 3e09c06f { memb(r9+#0) |= r15
memb(r9)|=r15

#CHECK: a157d100 { memh(r23+#0) = r17
memh(r23)=r17

#CHECK: a10fd400 { memb(r15+#0) = r20
memb(r15)=r20

#CHECK: 9082c014 { r21:20 = memb_fifo(r2+#0)
r21:20=memb_fifo(r2)

#CHECK: 9056c01c { r29:28 = memh_fifo(r22+#0)
r29:28=memh_fifo(r22)

#CHECK: a1d8ca00 { memd(r24+#0) = r11:10
memd(r24)=r11:10

#CHECK: a19ed900 { memw(r30+#0) = r25
memw(r30)=r25

#CHECK: a169ce00 { memh(r9+#0) = r14.h
memh(r9)=r14.h

#CHECK: 3f07c06b { memb(r7+#0) = setbit(#11)
memb(r7)=setbit(#11)

#CHECK: 3f34c07b { memh(r20+#0) = setbit(#27)
memh(r20)=setbit(#27)

#CHECK: 3f1cc032 { memb(r28+#0) -= #18
memb(r28)-=#18

#CHECK: 3f29c02a { memh(r9+#0) -= #10
memh(r9)-=#10

#CHECK: 3f4cc026 { memw(r12+#0) -= #6
memw(r12)-=#6

#CHECK: 3f00c00c { memb(r0+#0) += #12
memb(r0)+=#12

#CHECK: 3f50c07a { memw(r16+#0) = setbit(#26)
memw(r16)=setbit(#26)

#CHECK: 3f1fc05d { memb(r31+#0) = clrbit(#29)
memb(r31)=clrbit(#29)

#CHECK: 3f20c05e { memh(r0+#0) = clrbit(#30)
memh(r0)=clrbit(#30)

#CHECK: 3f42c059 { memw(r2+#0) = clrbit(#25)
memw(r2)=clrbit(#25)

#CHECK: 39cfe072 if (!p3.new) memw(r15+#0) = #-14
{
  p3=cmp.eq(r5,##-1997506977)
  if (!p3.new) memw(r15)=#-14
}

#CHECK: 3959e06b if (p3.new) memw(r25+#0) = #-21
{
  p3=cmp.eq(r0,##1863618461)
  if (p3.new) memw(r25)=#-21
}

#CHECK: 4312c801 if (p1.new) r1 = memb(r18+#0)
{
  if (p1.new) r1=memb(r18)
  p1=cmp.eq(r23,##-1105571618)
}

#CHECK: 4718d803 if (!p3.new) r3 = memb(r24+#0)
{
  if (!p3.new) r3=memb(r24)
  p3=cmp.eq(r3,##-210870878)
}

#CHECK: 4326c81b if (p1.new) r27 = memub(r6+#0)
{
  if (p1.new) r27=memub(r6)
  p1=cmp.eq(r29,##-188410493)
}

#CHECK: 473ad00d if (!p2.new) r13 = memub(r26+#0)
{
  p2=cmp.eq(r30,##-1823852150)
  if (!p2.new) r13=memub(r26)
}

#CHECK: 4785d80e if (!p3.new) r14 = memw(r5+#0)
{
  if (!p3.new) r14=memw(r5)
  p3=cmp.eq(r31,##-228524711)
}

#CHECK: 438cc81a if (p1.new) r26 = memw(r12+#0)
{
  if (p1.new) r26=memw(r12)
  p1=cmp.eq(r11,##-485232313)
}

#CHECK: 477dc019 if (!p0.new) r25 = memuh(r29+#0)
{
  p0=cmp.eq(r23,##127565957)
  if (!p0.new) r25=memuh(r29)
}

#CHECK: 4377c807 if (p1.new) r7 = memuh(r23+#0)
{
  p1=cmp.eq(r30,##-222020054)
  if (p1.new) r7=memuh(r23)
}

#CHECK: 4754c81c if (!p1.new) r28 = memh(r20+#0)
{
  p1=cmp.eq(r18,##1159699785)
  if (!p1.new) r28=memh(r20)
}

#CHECK: 435ec01b if (p0.new) r27 = memh(r30+#0)
{
  p0=cmp.eq(r7,##-1114567705)
  if (p0.new) r27=memh(r30)
}

#CHECK: 420dd100 if (p0.new) memb(r13+#0) = r17
{
  p0=cmp.eq(r21,##-1458796638)
  if (p0.new) memb(r13)=r17
}

#CHECK: 4601d602 if (!p2.new) memb(r1+#0) = r22
{
  p2=cmp.eq(r20,##-824022439)
  if (!p2.new) memb(r1)=r22
}

#CHECK: 43dcd808 if (p3.new) r9:8 = memd(r28+#0)
{
  p3=cmp.eq(r13,##56660744)
  if (p3.new) r9:8=memd(r28)
}

#CHECK: 47d8c80e if (!p1.new) r15:14 = memd(r24+#0)
{
  if (!p1.new) r15:14=memd(r24)
  p1=cmp.eq(r15,##1536716489)
}

#CHECK: 3918e045 if (p2.new) memb(r24+#0) = #-27
{
  if (p2.new) memb(r24)=#-27
  p2=cmp.eq(r21,##1741091811)
}

#CHECK: 398fe04d if (!p2.new) memb(r15+#0) = #-19
{
  if (!p2.new) memb(r15)=#-19
  p2=cmp.eq(r15,##779870261)
}

#CHECK: 3931c04b if (p2.new) memh(r17+#0) = #11
{
  if (p2.new) memh(r17)=#11
  p2=cmp.eq(r13,##-1171145798)
}

#CHECK: 39aee056 if (!p2.new) memh(r14+#0) = #-10
{
  p2=cmp.eq(r23,##-633976762)
  if (!p2.new) memh(r14)=#-10
}

#CHECK: 4692df01 if (!p1.new) memw(r18+#0) = r31
{
  if (!p1.new) memw(r18)=r31
  p1=cmp.eq(r11,##-319375732)
}

#CHECK: 428dc402 if (p2.new) memw(r13+#0) = r4
{
  if (p2.new) memw(r13)=r4
  p2=cmp.eq(r18,##1895120239)
}

#CHECK: 4670c300 if (!p0.new) memh(r16+#0) = r3.h
{
  p0=cmp.eq(r25,##1348715015)
  if (!p0.new) memh(r16)=r3.h
}

#CHECK: 426ddf02 if (p2.new) memh(r13+#0) = r31.h
{
  p2=cmp.eq(r25,##1085560657)
  if (p2.new) memh(r13)=r31.h
}

#CHECK: 464bcb01 if (!p1.new) memh(r11+#0) = r11
{
  p1=cmp.eq(r10,##1491455911)
  if (!p1.new) memh(r11)=r11
}

#CHECK: 4248d200 if (p0.new) memh(r8+#0) = r18
{
  p0=cmp.eq(r3,##687581160)
  if (p0.new) memh(r8)=r18
}

#CHECK: 42deca00 if (p0.new) memd(r30+#0) = r11:10
{
  if (p0.new) memd(r30)=r11:10
  p0=cmp.eq(r28,##562796189)
}

#CHECK: 46d5cc03 if (!p3.new) memd(r21+#0) = r13:12
{
  if (!p3.new) memd(r21)=r13:12
  p3=cmp.eq(r6,##-969273288)
}

#CHECK: 42bad201 if (p1.new) memw(r26+#0) = r22.new
{
  if (p1.new) memw(r26)=r22.new
  p1=cmp.eq(r0,##-1110065473)
  r22=add(r28,r9)
}

#CHECK: 46b9d201 if (!p1.new) memw(r25+#0) = r26.new
{
  p1=cmp.eq(r11,##-753121346)
  r26=add(r19,r7)
  if (!p1.new) memw(r25)=r26.new
}

#CHECK: 40aad200 if (p0) memw(r10+#0) = r6.new
{
  r6=add(r30,r0)
  if (p0) memw(r10)=r6.new
}

#CHECK: 44a6d202 if (!p2) memw(r6+#0) = r4.new
{
  if (!p2) memw(r6)=r4.new
  r4=add(r0,r3)
}

#CHECK: 40b9c200 if (p0) memb(r25+#0) = r29.new
{
  if (p0) memb(r25)=r29.new
  r29=add(r27,r30)
}

#CHECK: 44bec203 if (!p3) memb(r30+#0) = r8.new
{
  if (!p3) memb(r30)=r8.new
  r8=add(r24,r4)
}

#CHECK: 46aecc01 if (!p1.new) memh(r14+#0) = r13.new
{
  if (!p1.new) memh(r14)=r13.new
  r13=add(r21,r2)
  p1=cmp.eq(r3,##-1529345886)
}

#CHECK: 42bcca02 if (p2.new) memh(r28+#0) = r18.new
{
  p2=cmp.eq(r15,##2048545649)
  if (p2.new) memh(r28)=r18.new
  r18=add(r9,r3)
}

#CHECK: 46aac200 if (!p0.new) memb(r10+#0) = r30.new
{
  p0=cmp.eq(r21,##-1160401822)
  r30=add(r9,r22)
  if (!p0.new) memb(r10)=r30.new
}

#CHECK: 42b8c202 if (p2.new) memb(r24+#0) = r11.new
{
  if (p2.new) memb(r24)=r11.new
  p2=cmp.eq(r30,##1267977346)
  r11=add(r8,r18)
}

#CHECK: 44a3ca00 if (!p0) memh(r3+#0) = r28.new
{
  r28=add(r16,r11)
  if (!p0) memh(r3)=r28.new
}

#CHECK: 40abca03 if (p3) memh(r11+#0) = r24.new
{
  if (p3) memh(r11)=r24.new
  r24=add(r18,r19)
}

#CHECK: a1abd200 memw(r11+#0) = r5.new
{
  memw(r11)=r5.new
  r5=add(r0,r10)
}

#CHECK: a1a2ca00 memh(r2+#0) = r18.new
{
  r18=add(r27,r18)
  memh(r2)=r18.new
}

#CHECK: a1bac200 memb(r26+#0) = r15.new
{
  r15=add(r22,r17)
  memb(r26)=r15.new
}

#CHECK: d328ce1c { r29:28 = vsubub(r15:14,r9:8)
r29:28=vsubb(r15:14,r9:8)

#CHECK: 8c5ed60c { r12 = asr(r30,#22):rnd
r12=asrrnd(r30,#23)

#CHECK: ed1ec109 { r9 = mpyi(r30,r1)
r9=mpyui(r30,r1)

#CHECK: e010d787 { r7 = +mpyi(r16,#188)
r7=mpyi(r16,#188)

#CHECK: d206eea2 { p2 = boundscheck(r7:6,r15:14):raw:hi
p2=boundscheck(r7,r15:14)

#CHECK: f27ac102 { p2 = cmp.gtu(r26,r1)
p2=cmp.ltu(r1,r26)

#CHECK: f240df00 { p0 = cmp.gt(r0,r31)
p0=cmp.lt(r31,r0)

#CHECK: 7586cc01 { p1 = cmp.gtu(r6,#96)
p1=cmp.geu(r6,#97)

#CHECK: 755dc9a2 { p2 = cmp.gt(r29,#77)
p2=cmp.ge(r29,#78)

#CHECK: d310d60a { r11:10 = vaddub(r17:16,r23:22)
r11:10=vaddb(r17:16,r23:22)

#CHECK: 8753d1e6 { r6 = tableidxh(r19,#7,#17):raw
r6=tableidxh(r19,#7,#18)

#CHECK: 8786d277 { r23 = tableidxw(r6,#3,#18):raw
r23=tableidxw(r6,#3,#20)

#CHECK: 7c4dfff8 { r25:24 = combine(#-1,#-101)
r25:24=#-101

#CHECK: 8866c09a { r26 = vasrhub(r7:6,#0):raw
r26=vasrhub(r7:6,#1):rnd:sat

#CHECK: 7654c016 { r22 = sub(#0,r20)
r22=neg(r20)

#CHECK: 802cc808 { r9:8 = vasrh(r13:12,#8):raw
r9:8=vasrh(r13:12,#9):rnd

#CHECK: 7614dfe5 { r5 = {{zxtb\(r20\)|and\(r20,#255\)}}
r5=zxtb(r20)

#CHECK: 00ab68e2 immext(#179976320)
#CHECK: 7500c500 p0 = cmp.eq(r0,##179976360)
{
	if (p0.new) r11=r26
	p0=cmp.eq(r0,##179976360)
}

#CHECK: 74f9c00f { if (!p3) r15 = add(r25,#0)
if (!p3) r15=r25

#CHECK: 7425c005 { if (p1) r5 = add(r5,#0)
if (p1) r5=r5

#CHECK: e9badae2 { r2 = vrcmpys(r27:26,r27:26):<<1:rnd:sat:raw:lo
r2=vrcmpys(r27:26,r26):<<1:rnd:sat

#CHECK: fd13f20e if (p0.new) r15:14 = {{r19:18|combine\(r19,r18\)}}
{
  p0=cmp.eq(r26,##1766934387)
  if (p0.new) r15:14=r19:18
}

#CHECK: fd07c6c2 { if (!p2) r3:2 = {{r7:6|combine\(r7,r6\)}}
if (!p2) r3:2=r7:6

#CHECK: fd0dcc7e { if (p3) r31:30 = {{r13:12|combine\(r13,r12\)}}
if (p3) r31:30=r13:12

#CHECK: 748ae015 if (!p0.new) r21 = add(r10,#0)
{
  p0=cmp.eq(r23,##805633208)
  if (!p0.new) r21=r10
}

#CHECK: d36ec6c8 { r9:8 = add(r15:14,r7:6):raw:lo
r9:8=add(r14,r7:6)

#CHECK: 01e65477 immext(#509943232)
#CHECK: 7516c3a3 p3 = cmp.eq(r22,##509943261)
{
  if (!p3.new) r9:8 = r25:24
  p3=cmp.eq(r22,##509943261)
}

#CHECK: 87e0d5e5 { r5 = tableidxd(r0,#15,#21):raw
r5=tableidxd(r0,#15,#24)

#CHECK: 8701db65 { r5 = tableidxb(r1,#3,#27):raw
r5=tableidxb(r1,#3,#27)

#CHECK: 767affe3 { r3 = sub(#-1,r26)
r3=not(r26)

#CHECK: f51ddc06 { r7:6 = {{r29:28|combine\(r29,r28\)}}
r7:6=r29:28

#CHECK: 9406c000 { dcfetch(r6+#0)
dcfetch(r6)

#CHECK: 6b20c001 { p1 = or(p0,p0)
p1=p0

#CHECK: eafcdc82 { r3:2 += vrcmpys(r29:28,r29:28):<<1:sat:raw:lo
r3:2+=vrcmpys(r29:28,r28):<<1:sat

#CHECK: e8ead092 { r19:18 = vrcmpys(r11:10,r17:16):<<1:sat:raw:lo
r19:18=vrcmpys(r11:10,r16):<<1:sat

#CHECK: 9082c014 { r21:20 = memb_fifo(r2+#0)
r21:20=memb_fifo(r2)

#CHECK: 9056c01c { r29:28 = memh_fifo(r22+#0)
r29:28=memh_fifo(r22)
