# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# Check that the value of the registers preserved in subprocess mode while
# making the ioctl system call are actually preserved correctly.

# LLVM-EXEGESIS-DEFREG RAX 3
# LLVM-EXEGESIS-DEFREG RCX 5
# LLVM-EXEGESIS-DEFREG RDX 7
# LLVM-EXEGESIS-DEFREG RBX B
# LLVM-EXEGESIS-DEFREG RSI D
# LLVM-EXEGESIS-DEFREG RDI 11
# LLVM-EXEGESIS-DEFREG RSP 13
# LLVM-EXEGESIS-DEFREG RBP 17
# LLVM-EXEGESIS-DEFREG R8 1D
# LLVM-EXEGESIS-DEFREG R9 1F
# LLVM-EXEGESIS-DEFREG R10 29
# LLVM-EXEGESIS-DEFREG R11 2B
# LLVM-EXEGESIS-DEFREG R12 2F
# LLVM-EXEGESIS-DEFREG R13 35
# LLVM-EXEGESIS-DEFREG R14 127
# LLVM-EXEGESIS-DEFREG R15 0

cmpq $0x3, %rax
cmovneq %r14, %r15
cmpq $0x5, %rcx
cmovneq %r14, %r15
cmpq $0x7, %rdx
cmovneq %r14, %r15
cmpq $0xB, %rbx
cmovneq %r14, %r15
cmpq $0xD, %rsi
cmovneq %r14, %r15
cmpq $0x11, %rdi
cmovneq %r14, %r15
cmpq $0x13, %rsp
cmovneq %r14, %r15
cmpq $0x17, %rbp
cmovneq %r14, %r15
cmpq $0x1d, %r8
cmovneq %r14, %r15
cmpq $0x1f, %r9
cmovneq %r14, %r15
cmpq $0x29, %r10
cmovneq %r14, %r15
cmpq $0x2b, %r11
cmovneq %r14, %r15
cmpq $0x2f, %r12
cmovneq %r14, %r15
cmpq $0x35, %r13
cmovneq %r14, %r15

movq $60, %rax
movq %r15, %rdi
syscall

# CHECK-NOT: error:           'Child benchmarking process exited with non-zero exit code: Child process returned with unknown exit code'

