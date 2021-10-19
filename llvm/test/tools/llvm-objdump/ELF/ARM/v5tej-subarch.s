@ RUN: llvm-mc < %s -triple armv5tej-elf -filetype=obj | llvm-objdump -d - | FileCheck %s

bxj:
bxj r0

@ CHECK-LABEL: bxj
@ CHECK: 20 ff 2f e1 bxj r0
