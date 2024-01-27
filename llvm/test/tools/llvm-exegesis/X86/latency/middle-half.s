# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# Check that we can use the middle-half repetition mode without crashing

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -opcode-name=ADD64rr -repetition-mode=middle-half-duplicate | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -opcode-name=ADD64rr -repetition-mode=middle-half-loop | FileCheck %s

# CHECK: - { key: latency, value: {{[0-9.]*}}, per_snippet_value: {{[0-9.]*}}
