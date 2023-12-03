# REQUIRES: exegesis-can-execute-x86_64, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency -snippets-file=%s -execution-mode=subprocess --use-dummy-perf-counters | FileCheck %s

# LLVM-EXEGESIS-DEFREG RAX 0

movq $5, %rax

# CHECK: measurements:   []
