# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: not llvm-exegesis -mtriple=x86_64-unknown-unknown -snippets-file=%s -mode=latency 2>&1 | FileCheck %s

# CHECK: llvm-exegesis error: Memory and snippet address annotations are only supported in subprocess execution mode

# LLVM-EXEGESIS-SNIPPET-ADDRESS 10000

movq $0, %rax
