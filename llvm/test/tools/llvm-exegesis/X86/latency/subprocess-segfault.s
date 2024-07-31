# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# CHECK: error: The snippet encountered a segmentation fault at address 10

# LLVM-EXEGESIS-DEFREG RBX 10
movq (%rbx), %rax
