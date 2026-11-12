# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# Check that the code is loaded in at the expected address.

# LLVM-EXEGESIS-SNIPPET-ADDRESS 20000
# LLVM-EXEGESIS-DEFREG RAX 0
# LLVM-EXEGESIS-DEFREG R14 127
# LLVM-EXEGESIS-DEFREG R15 0
# LLVM-EXEGESIS-DEFREG RDI 0

# Load the instruction pointer and round down to the nearest page as there
# will be some setup code loaded in before this part begins to execute.
lea 0(%rip), %rax
shrq $12, %rax
shlq $12, %rax

cmpq $0x20000, %rax
cmovneq %r14, %r15

movq $60, %rax
movq %r15, %rdi
syscall

# CHECK-NOT: error:           'Child benchmarking process exited with non-zero exit code: Child process returned with unknown exit code'
