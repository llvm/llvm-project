// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s

lea rax, qword ptr [rip + .L.str]
