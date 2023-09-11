# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv62 -filetype=obj -o - %s | llvm-objdump --triple=hexagon --triple=hexagon --mcpu=hexagonv62 -d - | FileCheck %s

  r31:30=vabsdiffb(r29:28, r27:26)
# CHECK: e8fadc1e { r31:30 = vabsdiffb(r29:28,r27:26)

  r25:24=vabsdiffub(r23:22, r21:20)
# CHECK: e8b4d618 { r25:24 = vabsdiffub(r23:22,r21:20)

  r19:18,p3=vminub(r17:16, r15:14)
# CHECK: eaeed072 { r19:18,p3 = vminub(r17:16,r15:14)

  r13:12=vtrunehb(r11:10, r9:8)
# CHECK: c18ac86c { r13:12 = vtrunehb(r11:10,r9:8)

  r7:6=vtrunohb(r5:4, r3:2)
# CHECK: c184c2a6 { r7:6 = vtrunohb(r5:4,r3:2)

  r1:0=vsplatb(r31)
# CHECK: 845fc080 { r1:0 = vsplatb(r31)
