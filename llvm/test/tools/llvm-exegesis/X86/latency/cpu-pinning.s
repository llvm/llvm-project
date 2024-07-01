# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -opcode-name=ADD64rr -execution-mode=subprocess | FileCheck %s

# CHECK: - { key: latency, value: {{[0-9.]*}}, per_snippet_value: {{[0-9.]*}}
