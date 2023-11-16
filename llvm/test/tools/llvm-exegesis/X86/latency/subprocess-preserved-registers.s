# REQUIRES: exegesis-can-execute-x86_64

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# See comment in ./subprocess-abnormal-exit-code.s on the transient
# PTRACE_ATTACH failure.
# ALLOW_RETRIES: 2

# Check that the value of the registers preserved in subprocess mode while
# making the ioctl system call are actually preserved correctly.

# LLVM-EXEGESIS-DEFREG RAX 11
# LLVM-EXEGESIS-DEFREG RDI 13
# LLVM-EXEGESIS-DEFREG RSI 17
# LLVM-EXEGESIS-DEFREG R13 0
# LLVM-EXEGESIS-DEFREG R12 127

cmpq $0x11, %rax
cmovneq %r12, %r13
cmpq $0x13, %rdi
cmovneq %r12, %r13
cmpq $0x17, %rsi
cmovneq %r12, %r13

movq $60, %rax
movq %r13, %rdi
syscall

# CHECK-NOT: error:           'Child benchmarking process exited with non-zero exit code: Child process returned with unknown exit code'

