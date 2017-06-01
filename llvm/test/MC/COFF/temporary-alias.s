// RUN: llvm-mc -triple=i686-pc-windows -filetype=obj -o %t %s
// RUN: llvm-objdump -d -r %t | FileCheck %s

.globl _main
_main:
// CHECK: 00 00 00 00
// CHECK-NEXT: 00000002:  IMAGE_REL_I386_DIR32 .rdata
movb L_alias1(%eax), %al
// CHECK: 01 00 00 00
// CHECK-NEXT: 00000008:  IMAGE_REL_I386_DIR32 .rdata
movb L_alias2(%eax), %al
retl

.section .rdata,"dr"
L_sym1:
.ascii "\001"
L_sym2:
.ascii "\002"

L_alias1 = L_sym1
L_alias2 = L_sym2
