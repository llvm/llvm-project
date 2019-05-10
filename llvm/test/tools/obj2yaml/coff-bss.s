# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t.obj
# RUN: llvm-objdump -h %t.obj | FileCheck %s
# RUN: obj2yaml %t.obj | yaml2obj -o %t.2.obj
# RUN: llvm-objdump -h %t.2.obj | FileCheck %s

# CHECK: Idx Name          Size     VMA          Type
# CHECK:     .bss          00000004 0000000000000000 BSS

# Before PR41836, Size would be 0 after yaml conversion.

.bss
.global gv_bss
gv_bss:
.long 0
