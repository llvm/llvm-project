# RUN: llvm-mc -triple s390x-linux-gnu -filetype=obj %s | \
# RUN: llvm-objdump --mcpu=z14 -d - | FileCheck %s

# Test the .insn directive for vector instructions.

#CHECK: e7 23 2f ff 10 13     vgef    %v2, 4095(%v3,%r2), 1
  .insn vrv,0xe70000000013,%v2,4095(%v3,%r2),1

#CHECK: e7 56 ff f1 20 4a     vftci   %v5, %v6, 4095, 2, 1
  .insn vri,0xe7000000004a,%v5,%v6,4095,2,1

#CHECK: e7 20 2f ff 30 06     vl      %v2, 4095(%r2), 3
  .insn vrx,0xe70000000006,%v2,4095(%r2),3

#CHECK: e7 16 00 01 00 21     vlgvb   %r1, %v6, 1
  .insn vrs,0xe70000003021,%r1,%v6,1(%r0),0
#CHECK: e7 16 00 00 30 21     vlgvg   %r1, %v6, 0
  .insn vrs,0xe70000003021,%r1,%v6,0(%r0),3

#CHECK: e7 37 00 00 00 56     vlr     %v3, %v7
  .insn vrr,0xe70000000056,%v3,%v7,0,0,0,0
#CHECK: e7 37 60 18 30 eb     wfchdbs %f3, %f7, %f6
  .insn vrr,0xe700000000eb,%v3,%v7,%v6,3,8,1

#CHECK: e6 0c 20 0c 01 35     vlrl    %v16, 12(%r2), 12
  .insn vsi,0xe60000000035,%v16,12(%r2),12

#CHECK: e7 01 00 00 0c 56     vlr     %v16, %v17
 .insn vrr,0xe70000000056,16,17,0,0,0,0
