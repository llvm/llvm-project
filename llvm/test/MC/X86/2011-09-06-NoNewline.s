// RUN: llvm-mc -triple i386-unknown-unknown %s
// PR10869
movl %gs:8, %eax