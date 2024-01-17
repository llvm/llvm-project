# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# Check that we can set the value of RSP in subprocess mode without
# segfaulting as we need to restore it after the rest of the setup is
# complete to prevent loading from the stack where we set it instead
# of where the stack actuall is.

# LLVM-EXEGESIS-MEM-DEF test1 4096 2147483647
# LLVM-EXEGESIS-MEM-MAP test1 1048576
# LLVM-EXEGESIS-DEFREG RAX 100000
# LLVM-EXEGESIS-DEFREG R14 100000
# LLVM-EXEGESIS-DEFREG RSP 100000

movq %r14, (%rax)

# CHECK-NOT: error:           'The benchmarking subprocess sent unexpected signal: Segmentation fault'
