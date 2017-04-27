// RUN: llvm-mc -triple x86_64-unknown-unknown %s

.intel_syntax
add rbx, 0B0h
add rbx, 0b0h
add rax, 0A0h
add rax, 0a0h
