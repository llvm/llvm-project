#RUN: llvm-mc -arch=hexagon -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

# Check encoding bits for GP-relative loads.

#CHECK: 4fc6ff8c { r13:12 = memd(gp+#421856) }
r13:12 = memd(gp+#421856)
#CHECK: 4fc6ff8c { r13:12 = memd(gp+#421856) }
r13:12 = memd(#421856)

#CHECK: 4d1ac4d2 { r18 = memb(gp+#46118) }
r18 = memb(gp+#46118)
#CHECK: 4d1ac4d2 { r18 = memb(gp+#46118) }
r18 = memb(#46118)

#CHECK: 4d81f772 { r18 = memw(gp+#134892) }
r18 = memw(gp+#134892)
#CHECK: 4d81f772 { r18 = memw(gp+#134892) }
r18 = memw(#134892)

#CHECK: 497de287 { r7 = memuh(gp+#30248) }
r7 = memuh(gp+#30248)
#CHECK: 497de287 { r7 = memuh(gp+#30248) }
r7 = memuh(#30248)

#CHECK: 4b43e87a { r26 = memh(gp+#36486) }
r26 = memh(gp+#36486)
#CHECK: 4b43e87a { r26 = memh(gp+#36486) }
r26 = memh(#36486)

#CHECK: 4f37d07f { r31 = memub(gp+#61059) }
r31 = memub(gp+#61059)
#CHECK: 4f37d07f { r31 = memub(gp+#61059) }
r31 = memub(#61059)
