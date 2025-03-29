# REQUIRES: x86-registered-target
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump --no-print-imm-hex -d -mllvm --x86-asm-syntax=intel %t | FileCheck %s
# RUN: llvm-objdump --no-print-imm-hex -d -mllvm=--x86-asm-syntax=intel %t | FileCheck %s

# CHECK: lea rax, [rsi + 4*rdi + 5]

  leaq 5(%rsi,%rdi,4), %rax
