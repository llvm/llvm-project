# RUN: llvm-mc -triple=x86_64 -filetype=obj %s -o - | llvm-objdump -d - | FileCheck %s

# CHECK: 0:       ff ff  <unknown>
.word 0xffff
