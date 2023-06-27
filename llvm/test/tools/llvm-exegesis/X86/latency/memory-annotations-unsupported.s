# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: not llvm-exegesis -mtriple=x86_64-unknown-unknown -snippets-file=%s -mode=latency 2>&1 | FileCheck %s

# CHECK: llvm-exegesis error: Memory annotations are only supported in subprocess execution mode

# LLVM-EXEGESIS-MEM-DEF test1 4096 ff

movq $0, %rax
