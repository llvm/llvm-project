# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# Test that specifying the loop register to use works as expected.

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s | FileCheck %s

# CHECK: measurements:

# LLVM-EXEGESIS-DEFREG R11 ff
# LLVM-EXEGESIS-LOOP-REGISTER R12

addq $0xff, %r11
