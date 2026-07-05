# RUN: llvm-mc -triple i386 %s -filetype=obj | llvm-objdump -d - | FileCheck %s

.nops 4
# CHECK:       0: 90 nop
# CHECK-NEXT:  1: 90 nop
# CHECK-NEXT:  2: 90 nop
# CHECK-NEXT:  3: 90 nop
.nops 4, 1
# CHECK:       4: 90 nop
# CHECK-NEXT:  5: 90 nop
# CHECK-NEXT:  6: 90 nop
# CHECK-NEXT:  7: 90 nop
