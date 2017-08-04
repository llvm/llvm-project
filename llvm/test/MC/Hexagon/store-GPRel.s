#RUN: llvm-mc -arch=hexagon -filetype=obj %s | llvm-objdump -d -r - | FileCheck %s

# Check encoding bits for gp-rel stores.

#CHECK: 4ab3f229 memw(gp+#105636) = r12.new
{ r12 = add(r0,r19)
  memw(gp+#105636) = r12.new }

#CHECK: 4ab3f229 memw(gp+#105636) = r12.new
{ r12 = add(r0,r19)
  memw(#105636) = r12.new }

#CHECK: 4ebdca35 memh(gp+#128106) = r6.new
{ r6 = add(r18,r13)
  memh(gp+#128106) = r6.new }

#CHECK: 4ebdca35 memh(gp+#128106) = r6.new
{ r6 = add(r18,r13)
  memh(#128106) = r6.new }

#CHECK: 4eb3e2fc memb(gp+#59388) = r17.new
{ r17 = add(r26,r18)
  memb(gp+#59388) = r17.new }
#CHECK: 4eb3e2fc memb(gp+#59388) = r17.new
{ r17 = add(r26,r18)
  memb(#59388) = r17.new }

#CHECK: 4ad2ea01 { memd(gp+#206856) = r11:10
{ memd(gp+#206856) = r11:10 }
#CHECK: 4ad2ea01 { memd(gp+#206856) = r11:10
{ memd(#206856) = r11:10 }

#CHECK: 4c9dfa1e { memw(gp+#191608) = r26
{ memw(gp+#191608) = r26 }
#CHECK: 4c9dfa1e { memw(gp+#191608) = r26
{ memw(#191608) = r26 }

#CHECK: 4855cfdc { memh(gp+#21944) = r15
{ memh(gp+#21944) = r15 }
#CHECK: 4855cfdc { memh(gp+#21944) = r15
{ memh(#21944) = r15 }

#CHECK: 4a00cea2 { memb(gp+#16546) = r14
{ memb(gp+#16546) = r14 }
#CHECK: 4a00cea2 { memb(gp+#16546) = r14
{ memb(#16546) = r14 }
