// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o - \
// RUN:   | llvm-readobj --unwind - | FileCheck %s

// CHECK:          Prologue [
// CHECK-NEXT:        0xdc01              ; str d8, [sp, #8]
// CHECK-NEXT:        0xd400              ; str x19, [sp, #-8]!
// CHECK-NEXT:        0xe4                ; end
// CHECK-NEXT:     ]

.section .pdata,"dr"
        .long func@IMGREL
        .long "$unwind$func"@IMGREL

        .text
        .globl  func
func:
        str x19, [sp, #-8]!
        str d8,  [sp, #8]
        ret

.section .xdata,"dr"
"$unwind$func":
.long 0x10000002, 0x00d401dc, 0xe3e3e3e4
