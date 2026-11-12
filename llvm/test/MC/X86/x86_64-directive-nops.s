# RUN: llvm-mc -triple=x86_64 %s -filetype=obj | llvm-objdump -d - | FileCheck %s

.nops 4, 1
# CHECK:       0: 90 nop
# CHECK-NEXT:  1: 90 nop
# CHECK-NEXT:  2: 90 nop
# CHECK-NEXT:  3: 90 nop
.nops 4, 2
# CHECK-NEXT:  4: 66 90 nop
# CHECK-NEXT:  6: 66 90 nop
.nops 4, 3
# CHECK-NEXT:  8: 0f 1f 00 nopl (%rax)
# CHECK-NEXT:  b: 90 nop
.nops 4, 4
# CHECK-NEXT:  c: 0f 1f 40 00 nopl (%rax)
.nops 4, 5
# CHECK-NEXT:  10: 0f 1f 40 00 nopl (%rax)
.nops 4
# CHECK-NEXT:  14: 0f 1f 40 00 nopl (%rax)
