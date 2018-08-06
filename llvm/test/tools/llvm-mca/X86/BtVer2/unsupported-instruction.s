# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck %s

bzhi %eax, %ebx, %ecx

# CHECK: error: found an unsupported instruction in the input assembly sequence.
# CHECK-NEXT: note: instruction: 	bzhil	%eax, %ebx, %ecx
