# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=generic %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=i386 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=i486 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=i586 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=pentium %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=pentium-mmx %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=geode %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=i686 %s | llvm-objdump -d - | not FileCheck %s

# CHECK-NOT: nop{{[lw]}}
inc %eax
.align 8
inc %eax
