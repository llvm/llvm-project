# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: not llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess -use-dummy-perf-counters 2>&1 | FileCheck %s

# CHECK: llvm-exegesis error: Dummy perf counters are not supported in the subprocess execution mode.

mov $0, %rax
