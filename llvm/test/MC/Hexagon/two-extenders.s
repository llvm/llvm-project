# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s
#

# In packets with two extensions assembler is not extending both instructions
#

//['D_DUMMY,C4_or_or,L4_ploadrbtnew_abs,S2_storerfgp']
{
        if (p3) r23 = memb(##2164335510)
        memh(##1696682668) = r28.h
}
# CHECK: { immext(#2164335488)
# CHECK:   if (p3) r23 = memb(##2164335510)
# CHECK:   immext(#1696682624)
# CHECK:   memh(##1696682668) = r28.h }

//['D_DUMMY,C4_or_or,L4_ploadrbtnew_abs,S2_storerfgp']
{
        if (p3.new) r23 = memb(##2164335510)
        p3 = or(p2,or(p3, p0))
}
# CHECK: { p3 = or(p2,or(p3,p0))
# CHECK:   immext(#2164335488)
# CHECK:   if (p3.new) r23 = memb(##2164335510) }


# -------------------------- Non-extended cases:
# -------------------------- Use GP and non GP notation

R2 = memb(gp+#0x1000)
# CHECK: { r2 = memb(gp+#4096) }

R3 = memh(gp+#0x1000)
# CHECK: { r3 = memh(gp+#4096) }

r4 = memub(gp+#0x1000)
# CHECK: { r4 = memub(gp+#4096) }

r5 = memuh(gp+#0x1000)
# CHECK: { r5 = memuh(gp+#4096) }

r6 = memw(gp+#0x1000)
# CHECK: { r6 = memw(gp+#4096) }

R1:0 = memd(gp+#0x1000)
# CHECK: { r1:0 = memd(gp+#4096) }

{R25 = #1; memb(gp+#0x1000) = R25.new}
# CHECK: { r25 = #1
# CHECK-NEXT: memb(gp+#4096) = r25.new }

{R26 = #1; memh(gp+#0x1000) = R26.new}
# CHECK: { r26 = #1
# CHECK-NEXT: memh(gp+#4096) = r26.new }

{R27 = #1; memw(gp+#0x1000) = R27.new}
# CHECK: { r27 = #1
# CHECK-NEXT: memw(gp+#4096) = r27.new }

memd(gp+#0x1000) = R1:0
# CHECK: { memd(gp+#4096) = r1:0 }

memb(gp+#0x1000) = R2
# CHECK: { memb(gp+#4096) = r2 }

memh(gp+#0x1000) = r3.h
# CHECK: { memh(gp+#4096) = r3.h }

memh(gp+#0x1000) = R4
# CHECK: { memh(gp+#4096) = r4 }

memw(gp+#0x1000) = R5
# CHECK: { memw(gp+#4096) = r5 }

# -------------------------- Extended cases:
# -------------------------- Use GP and non GP notation

R11:10 = memd(##0x1000)
# CHECK: { immext(#4096)
# CHECK-NEXT: r11:10 = memd(##4096) }

R11 = memb(##0x1000)
# CHECK: { immext(#4096)
# CHECK-NEXT: r11 = memb(##4096) }

R12 = memh(##0x1000)
# CHECK: { immext(#4096)
# CHECK-NEXT: r12 = memh(##4096) }

r13 = memub(##0x1000)
# CHECK: { immext(#4096)
# CHECK-NEXT: r13 = memub(##4096) }

r14 = memuh(##0x1000)
# CHECK: { immext(#4096)
# CHECK-NEXT: r14 = memuh(##4096) }

r15 = memw(##0x1000)
# CHECK: { immext(#4096)
# CHECK-NEXT: r15 = memw(##4096) }

{R22 = #1; memb(##0x1000) = R22.new}
# CHECK: { r22 = #1
# CHECK-NEXT: immext(#4096)
# CHECK-NEXT: memb(##4096) = r22.new }

{R23 = #1; memh(##0x1000) = R23.new}
# CHECK: { r23 = #1
# CHECK-NEXT: immext(#4096)
# CHECK-NEXT: memh(##4096) = r23.new }

{R24 = #1; memw(##0x1000) = R24.new}
# CHECK: { r24 = #1
# CHECK-NEXT: immext(#4096)
# CHECK-NEXT: memw(##4096) = r24.new }

memd(##0x1000) = R17:16
# CHECK: { immext(#4096)
# CHECK-NEXT: memd(##4096) = r17:16 }

memb(##0x1000) = R18
# CHECK: { immext(#4096)
# CHECK-NEXT: memb(##4096) = r18 }

memh(##0x1000) = r19.h
# CHECK: { immext(#4096)
# CHECK-NEXT: memh(##4096) = r19.h }

memh(##0x1000) = R20
# CHECK: { immext(#4096)
# CHECK-NEXT: memh(##4096) = r20 }

memw(##0x1000) = R21
# CHECK: { immext(#4096)
# CHECK-NEXT: memw(##4096) = r21 }
