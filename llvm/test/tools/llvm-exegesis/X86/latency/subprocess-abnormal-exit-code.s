# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: not llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess 2>&1 | FileCheck %s

# CHECK: llvm-exegesis error: Child benchmarking process exited with non-zero exit code: Child process returned with unknown exit code

movl $60, %eax
movl $127, %edi
syscall
