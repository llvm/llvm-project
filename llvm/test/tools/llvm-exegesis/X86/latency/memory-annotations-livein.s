# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# Test that we can use the subprocess executor mode with memory annotations
# while having live-ins still work as expected.

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# CHECK: measurements:
# CHECK-NEXT: value: {{.*}}, per_snippet_value: {{.*}}

# TODO: Sometimes transiently fails on PTRACE_ATTACH
# ALLOW_RETRIES: 2

# LLVM-EXEGESIS-MEM-DEF test1 4096 2147483647
# LLVM-EXEGESIS-MEM-MAP test1 1048576
# LLVM-EXEGESIS-LIVEIN R14

movq $1048576, %rax
movq %r14, (%rax)
