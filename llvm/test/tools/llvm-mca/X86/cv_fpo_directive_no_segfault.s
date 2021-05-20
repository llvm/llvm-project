# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=generic -resource-pressure=false -instruction-info=false < %s | FileCheck %s

.cv_fpo_pushreg ebx
add %eax, %eax
add %ebx, %ebx
add %ecx, %ecx
add %edx, %edx

# CHECK:      Iterations:        100
