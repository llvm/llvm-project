# RUN: llvm-mc -triple=hexagon --mcpu=hexagonv65 -filetype=obj %s | \
# RUN:     llvm-objdump --mcpu=hexagonv65 -d - | FileCheck %s
# This checks correct encoding of system and cache instructions.

#CHECK: a800c000 { barrier }
barrier

#CHECK: a200c000 { dckill }
dckill

#CHECK: a820c000 { l2kill }
l2kill

#CHECK: a840c000 { syncht }
syncht

#CHECK: a00ac000 { dccleana(r10) }
dccleana(r10)

#CHECK: a02bc000 { dcinva(r11) }
dcinva(r11)

#CHECK: a04cc000 { dccleaninva(r12) }
dccleaninva(r12)

#CHECK: a22dc000 { dccleanidx(r13) }
dccleanidx(r13)

#CHECK: a410d100 { dctagw(r16,r17) }
dctagw(r16,r17)

#CHECK: a24ec000 { dcinvidx(r14) }
dcinvidx(r14)

#CHECK: a26fc000 { dccleaninvidx(r15) }
dccleaninvidx(r15)

#CHECK: a433c012 { r18 = dctagr(r19) }
r18=dctagr(r19)

#CHECK: a454d500 { l2tagw(r20,r21) }
l2tagw(r20,r21)

#CHECK: a477c016 { r22 = l2tagr(r23) }
r22=l2tagr(r23)

#CHECK: a618d900 { l2fetch(r24,r25) }
l2fetch(r24,r25)

#CHECK: a87ac000 { l2cleaninvidx(r26) }
l2cleaninvidx(r26)

#CHECK: 6401c000 { swi(r1) }
swi(r1)

#CHECK: 6402c020 { cswi(r2) }
cswi(r2)

#CHECK: 6403c040 { iassignw(r3) }
iassignw(r3)

#CHECK: 6404c060 { ciad(r4) }
ciad(r4)

#CHECK: 6445c000 { wait(r5) }
wait(r5)

#CHECK: 6446c020 { resume(r6) }
resume(r6)

#CHECK: 6467c000 { stop(r7) }
stop(r7)

#CHECK: 6468c020 { start(r8) }
start(r8)

#CHECK: 6469c040 { nmi(r9) }
nmi(r9)

#CHECK: 648ac060 { siad(r10) }
siad(r10)

#CHECK: 648bc300 { setimask(p3,r11) }
setimask(p3,r11)

#CHECK: 650cc000 { crswap(r12,sgp0) }
crswap(r12,sgp0)

#CHECK: 652dc000 { crswap(r13,sgp1) }
crswap(r13,sgp1)

#CHECK: 6d8ec000 { crswap(r15:14,s1:0) }
crswap(r15:14,sgp1:0)

#CHECK: 660fc00e { r14 = getimask(r15) }
r14=getimask(r15)

#CHECK: 6671c010 { r16 = iassignr(r17) }
r16=iassignr(r17)

#CHECK: 6700c006 { ssr = r0 }
ssr=r0

#CHECK: 6c0cc300 { tlbw(r13:12,r3) }
tlbw(r13:12,r3)

#CHECK: 6c20c000 { brkpt }
brkpt

#CHECK: 6c20c020 { tlblock }
tlblock

#CHECK: 6c20c040 { tlbunlock }
tlbunlock

#CHECK: 6c20c060 { k0lock }
k0lock

#CHECK: 6c20c080 { k0unlock }
k0unlock

#CHECK: 6c44c002 { r3:2 = tlbr(r4) }
r3:2=tlbr(R4)

#CHECK: 6c86c005 { r5 = tlbp(r6) }
r5=tlbp(r6)

#CHECK: 6d0ac03e { s63:62 = r11:10 }
s63:62=r11:10

#CHECK: 6e86c000 { r0 = ssr }
r0=ssr

#CHECK: 6f3ec000 { r1:0 = s63:62 }
r1:0=s63:62

#CHECK: 6a49c01f   r31 = add(pc,##0x400000) }
r31=add (pc,#0x400000)

#CHECK: 6a49c21f { r31 = add(pc,#0x4) }
r31=add (pc,#0x4)

#CHECK: 625fc000 { trace(r31) }
trace(r31)

#CHECK: 6ce6c00b { r11 = tlboc(r7:6) }
r11=tlboc(r7:6)

#CHECK: 6cbcc000 { tlbinvasid(r28) }
tlbinvasid(r28)

#CHECK: a656c000 { l2invidx(r22) }
l2invidx(r22)

#CHECK: a693c000 { l2fetch(r19,r1:0) }
l2fetch(r19,r1:0)

#CHECK: a634c000 { l2cleanidx(r20) }
l2cleanidx(r20)

#CHECK: 6cc6c60c { r12 = ctlbw(r7:6,r6) }
r12=ctlbw(r7:6,r6)

#CHECK: a618cd00 { l2fetch(r24,r13) }
l2fetch(r24,r13)

# SWI isSoloAX not isSolo
#CHECK-NOT: error:
{
  r0 = r0
  swi(r1)
}
