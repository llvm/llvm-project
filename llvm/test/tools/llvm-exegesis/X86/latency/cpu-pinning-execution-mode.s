# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: not llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -opcode-name=ADD64rr -execution-mode=inprocess --benchmark-process-cpu=0 2>&1 | FileCheck %s

# CHECK: llvm-exegesis error: --benchmark-process-cpu is only supported in the subprocess execution mode
