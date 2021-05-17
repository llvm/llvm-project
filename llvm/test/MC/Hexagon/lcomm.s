# RUN: llvm-mc -filetype=obj -triple=hexagon %s | llvm-objdump -t - | FileCheck %s

# CHECK:      00000000 l     O .sbss.1                 00000001 dst1
# CHECK-NEXT: 00000000 l    d  .sbss.1                 00000000 .sbss.1
# CHECK-NEXT: 00000000 l     O .sbss.2                 00000002 dst2
# CHECK-NEXT: 00000000 l    d  .sbss.2                 00000000 .sbss.2
# CHECK-NEXT: 00000000 l     O .sbss.4                 00000004 dst4
# CHECK-NEXT: 00000000 l    d  .sbss.4                 00000000 .sbss.4
# CHECK-NEXT: 00000000 l     O .sbss.8                 00000008 dst8
# CHECK-NEXT: 00000000 l    d  .sbss.8                 00000000 .sbss.8

.lcomm dst1,1,1,1
.lcomm dst2,2,2,2
.lcomm dst4,4,4,4
.lcomm dst8,8,8,8

r0 = add(pc, ##dst1@PCREL)
r0 = add(pc, ##dst2@PCREL)
r0 = add(pc, ##dst4@PCREL)
r0 = add(pc, ##dst8@PCREL)
