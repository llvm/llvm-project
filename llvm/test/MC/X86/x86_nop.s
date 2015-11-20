# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=generic %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=i386 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=i486 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=i586 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=pentium %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=pentium-mmx %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=geode %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=i686 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=k6 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=k6-2 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=k6-3 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=winchip-c6 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=winchip2 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=c3 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=c3-2 %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=core2 %s | llvm-objdump -d - | FileCheck --check-prefix=NOPL %s
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux -mcpu=slm %s | llvm-objdump -d - | FileCheck --check-prefix=NOPL %s


inc %eax
.align 8
inc %eax

// CHECK: 0:	40                                           	incl	%eax
// CHECK: 1:	8d b4 26 00 00 00 00                            leal (%esi), %esi
// CHECK: 8:	40                                           	incl	%eax


// NOPL: 0:	40                                           	incl	%eax
// NOPL: 1:	0f 1f 80 00 00 00 00                         	nopl	(%eax)
// NOPL: 8:	40                                           	incl	%eax
