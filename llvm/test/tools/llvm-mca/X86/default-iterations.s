# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 < %s 2>&1 | FileCheck --check-prefix=DEFAULT %s
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=0 < %s 2>&1 | FileCheck --check-prefix=DEFAULT %s
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=1 < %s 2>&1 | FileCheck --check-prefix=CUSTOM %s

add %eax, %eax

# DEFAULT: Iterations: 70
# DEFAULT-NEXT: Instructions: 70

# CUSTOM: Iterations: 1
# CUSTOM-NEXT: Instructions: 1
