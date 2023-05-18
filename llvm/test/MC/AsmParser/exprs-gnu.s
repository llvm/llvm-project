# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s

# CHECK: movl %eax, 15
movl %eax, 3 ! ~12
