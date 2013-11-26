# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=generic %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=i386 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=i486 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=i586 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=pentium %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=pentium-mmx %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=geode %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=i686 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=k6 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=k6-2 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=k6-3 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=winchip-c6 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=winchip2 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=c3 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=c3-2 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -arch=x86 -mcpu=core2 %s | llvm-objdump -d - | FileCheck --check-prefix=NOPL %s


inc %eax
.align 8
inc %eax

// CHECK: 0:	40                                           	incl	%eax
// CHECK: 1:	90                                           	nop
// CHECK: 2:	90                                           	nop
// CHECK: 3:	90                                           	nop
// CHECK: 4:	90                                           	nop
// CHECK: 5:	90                                           	nop
// CHECK: 6:	90                                           	nop
// CHECK: 7:	90                                           	nop
// CHECK: 8:	40                                           	incl	%eax


// NOPL: 0:	40                                           	incl	%eax
// NOPL: 1:	0f 1f 80 00 00 00 00                         	nopl	(%eax)
// NOPL: 8:	40                                           	incl	%eax
