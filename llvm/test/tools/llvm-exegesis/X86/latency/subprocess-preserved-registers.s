# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# Check that the value of the registers preserved in subprocess mode while
# making the ioctl system call are actually preserved correctly.

# LLVM-EXEGESIS-DEFREG RAX 3
# LLVM-EXEGESIS-DEFREG RCX 5
# LLVM-EXEGESIS-DEFREG RDI 7
# LLVM-EXEGESIS-DEFREG RSI B
# LLVM-EXEGESIS-DEFREG R11 D
# LLVM-EXEGESIS-DEFREG R14 127
# LLVM-EXEGESIS-DEFREG R15 0

cmpq $0x3, %rax
cmovneq %r14, %r15
cmpq $0x5, %rcx
cmovneq %r14, %r15
cmpq $0x7, %rdi
cmovneq %r14, %r15
cmpq $0xB, %rsi
cmovneq %r14, %r15
cmpq $0xD, %r11
cmovneq %r14, %r15

movq $60, %rax
movq %r15, %rdi
syscall

# CHECK-NOT: error:           'Child benchmarking process exited with non-zero exit code: Child process returned with unknown exit code'

