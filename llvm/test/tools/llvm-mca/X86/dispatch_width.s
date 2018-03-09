# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 < %s 2>&1 | FileCheck --check-prefix=DEFAULT %s
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -dispatch=0 < %s 2>&1 | FileCheck --check-prefix=DEFAULT %s
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -dispatch=1 < %s 2>&1 | FileCheck --check-prefix=CUSTOM %s

add %eax, %eax

# DEFAULT: Dispatch Width: 2
# CUSTOM: Dispatch Width: 1
