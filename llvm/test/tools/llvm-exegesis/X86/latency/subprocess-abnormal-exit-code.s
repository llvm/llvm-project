# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# CHECK: error: 'Child benchmarking process exited with non-zero exit code: Child process returned with unknown exit code'

# TODO: Sometimes transiently fails on PTRACE_ATTACH
# ALLOW_RETRIES: 2

movl $60, %eax
movl $127, %edi
syscall
