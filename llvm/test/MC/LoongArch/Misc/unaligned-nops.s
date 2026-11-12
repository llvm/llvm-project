# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK:      01 00 00 00   <unknown>
# CHECK-NEXT: 00 00 40 03   nop
.byte 1
.p2align 3
foo:
