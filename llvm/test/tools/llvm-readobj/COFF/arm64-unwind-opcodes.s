// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o %t.o
// RUN: llvm-readobj --unwind %t.o | FileCheck --strict-whitespace %s

// CHECK:          Prologue [
// CHECK-NEXT:        0xe202              ; add fp, sp, #16
// CHECK-NEXT:        0xe1                ; mov fp, sp
// CHECK-NEXT:        0xdc01              ; str d8, [sp, #8]
// CHECK-NEXT:        0xd400              ; str x19, [sp, #-8]!
// CHECK-NEXT:        0xe4                ; end
// CHECK-NEXT:     ]
// CHECK-NEXT:     EpilogueScopes [
// CHECK-NEXT:       EpilogueScope {
// CHECK-NEXT:         StartOffset:
// CHECK-NEXT:         EpilogueStartIndex:
// CHECK-NEXT:         Opcodes [
// CHECK-NEXT:           0xe202              ; sub sp, fp, #16
// CHECK-NEXT:           0xe1                ; mov sp, fp
// CHECK-NEXT:           0xe4                ; end
// CHECK-NEXT:         ]
// CHECK-NEXT:       }
// CHECK-NEXT:     ]

.section .pdata,"dr"
        .long func@IMGREL
        .long "$unwind$func"@IMGREL

        .text
        .globl  func
func:
        str x19, [sp, #-8]!
        str d8, [sp, #8]
        mov x29, sp
        add x29, sp, #16
        nop
        sub sp, x29, #16
        mov sp, x29
        ret

.section .xdata,"dr"
"$unwind$func":
.byte 0x08, 0x00, 0x40, 0x18
.byte 0x05, 0x00, 0x00, 0x02
.byte 0xe2, 0x02, 0xe1, 0xdc
.byte 0x01, 0xd4, 0x00, 0xe4
.byte 0xe2, 0x02, 0xe1, 0xe4
