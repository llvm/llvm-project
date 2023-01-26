// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o %t.o
// RUN: llvm-readobj --unwind %t.o | FileCheck --strict-whitespace %s

// CHECK:            Prologue [
// CHECK-NEXT:         0xd600              ; stp x19, lr, [sp, #0]
// CHECK-NEXT:         0x01                ; sub sp, #16
// CHECK-NEXT:         0xfc                ; pacibsp
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:       Epilogue [
// CHECK-NEXT:         0x01                ; add sp, #16
// CHECK-NEXT:         0xfc                ; autibsp
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]

.section .pdata,"dr"
        .long func@IMGREL
        .long "$unwind$func"@IMGREL

        .text
        .globl  func
func:
        pacibsp
        sub sp, sp, #16
        stp x19, x30, [sp]
        mov w19, w1
        blr x0
        mov w0, w19
        ldp x19, x30, [sp]
        add sp, sp, #16
        autibsp
        ret

.section .xdata,"dr"
"$unwind$func":
.byte 0x0a, 0x00, 0xa0, 0x10
.byte 0xd6, 0x00, 0x01, 0xfc
.byte 0xe4, 0xe3, 0xe3, 0xe3
