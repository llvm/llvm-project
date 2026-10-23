# RUN: not llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s 2>&1 | FileCheck %s

# CHECK: error: division by zero in assembly expression
mov eax, 1 / 0

# CHECK: error: modulo by zero in assembly expression
mov eax, 1 % 0
