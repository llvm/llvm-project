// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
// RUN: llvm-objdump -dr --no-print-imm-hex %t | FileCheck %s

// CHECK:      <_start>:
// CHECK-NEXT:   movq (%rip), %r16
// CHECK-NEXT:   R_X86_64_CODE_4_GOTTPOFF    tls0-0x4
// CHECK-NEXT:   movq (%rip), %r20
// CHECK-NEXT:   R_X86_64_CODE_4_GOTTPOFF    tls0-0x4
// CHECK-NEXT:   movq (%rip), %r16
// CHECK-NEXT:   R_X86_64_CODE_4_GOTTPOFF    tls1-0x4
// CHECK-NEXT:   addq (%rip), %r16
// CHECK-NEXT:   R_X86_64_CODE_4_GOTTPOFF    tls0-0x4
// CHECK-NEXT:   addq (%rip), %r28
// CHECK-NEXT:   R_X86_64_CODE_4_GOTTPOFF    tls0-0x4
// CHECK-NEXT:   addq (%rip), %r16
// CHECK-NEXT:   R_X86_64_CODE_4_GOTTPOFF    tls1-0x4

.type tls0,@object
.section .tbss,"awT",@nobits
.globl tls0
tls0:
.long 0
.type  tls1,@object
.globl tls1
tls1:
.long 0
.section .text
.globl _start
_start:
    # EGPR
    movq tls0@GOTTPOFF(%rip), %r16
    movq tls0@GOTTPOFF(%rip), %r20
    movq tls1@GOTTPOFF(%rip), %r16
    addq tls0@GOTTPOFF(%rip), %r16
    addq tls0@GOTTPOFF(%rip), %r28
    addq tls1@GOTTPOFF(%rip), %r16

