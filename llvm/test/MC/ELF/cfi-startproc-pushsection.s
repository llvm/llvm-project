// RUN: llvm-mc -filetype=obj -triple x86_64-linux %s -o - | llvm-dwarfdump --eh-frame - | FileCheck %s

.section .text.a, "ax", %progbits
.cfi_startproc
.cfi_def_cfa %rsp, 0

.pushsection .text.b, "ax", %progbits
.cfi_startproc simple
.cfi_def_cfa %rsp, 8
nop
ret

.pushsection .text.c, "ax", %progbits
.cfi_startproc simple
.cfi_def_cfa %rsp, 16
nop
nop
ret
.cfi_endproc
.popsection

.cfi_endproc
.popsection

.pushsection .text.d, "ax", %progbits
.cfi_startproc simple
.cfi_def_cfa %rsp, 24
nop
nop
nop
ret
.cfi_endproc
.popsection

ret
.cfi_endproc

// CHECK: pc=00000000...00000001
// CHECK: RSP +0
// CHECK: pc=00000000...00000002
// CHECK: RSP +8
// CHECK: pc=00000000...00000003
// CHECK: RSP +16
// CHECK: pc=00000000...00000004
// CHECK: RSP +24
