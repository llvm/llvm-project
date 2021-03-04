# RUN: llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=atom -o /dev/null 2>&1 | FileCheck %s
# CHECK: warning: support for in-order CPU 'atom' is experimental.
movsbw	%al, %di
