# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# CHECK: error:           'The benchmarking subprocess sent unexpected signal: Segmentation fault'

# TODO: Sometimes transiently fails on PTRACE_ATTACH
# ALLOW_RETRIES: 2

# LLVM-EXEGESIS-DEFREG RBX 0
movq (%rbx), %rax
