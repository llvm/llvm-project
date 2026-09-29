# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# Check that the value of the segment registers is set properly when in
# subprocess mode.

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# LLVM-EXEGESIS-DEFREG FS 12345600
# LLVM-EXEGESIS-DEFREG GS 2468ac00
# LLVM-EXEGESIS-DEFREG R13 0
# LLVM-EXEGESIS-DEFREG R14 127
# LLVM-EXEGESIS-DEFREG R15 0
# LLVM-EXEGESIS-MEM-DEF MEM1 4096 0000000012345600
# LLVM-EXEGESIS-MEM-DEF MEM2 4096 000000002468ac00
# LLVM-EXEGESIS-MEM-MAP MEM1 305418240
# LLVM-EXEGESIS-MEM-MAP MEM2 610836480

movq %fs:0, %r13
cmpq $0x12345600, %r13
cmovneq %r14, %r15
movq %gs:0, %r13
cmpq $0x2468ac00, %r13
cmovneq %r14, %r15

movq $60, %rax
movq %r15, %rdi
syscall

# CHECK-NOT: error:           'Child benchmarking process exited with non-zero exit code: Child process returned with unknown exit code'
