@ RUN: llvm-mc < %s -triple armv5tej-elf -filetype=obj | llvm-objdump -d - | FileCheck %s

bxj:
bxj r0

@ CHECK-LABEL: bxj
@ CHECK: e12fff20    bxj r0
