# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# CHECK: measurements:
# CHECK-NEXT: value: {{.*}}, per_snippet_value: {{.*}}

# TODO: Sometimes transiently fails on PTRACE_ATTACH
# ALLOW_RETRIES: 2

# LLVM-EXEGESIS-DEFREG XMM1 42
# LLVM-EXEGESIS-DEFREG XMM2 42
# LLVM-EXEGESIS-DEFREG XMM3 42
vhaddps       %xmm2, %xmm2, %xmm3
