// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o %t.o
// RUN: llvm-readobj --unwind %t.o | FileCheck --strict-whitespace %s

//CHECK:           Prologue [
//CHECK-NEXT:        0xe1                ; mov fp, sp
//CHECK-NEXT:        0x83                ; stp x29, x30, [sp, #-32]!
//CHECK-NEXT:        0xe6                ; save next
//CHECK-NEXT:        0xe6                ; save next
//CHECK-NEXT:        0xe6                ; save next
//CHECK-NEXT:        0xe6                ; save next
//CHECK-NEXT:        0xe76689            ; stp q6, q7, [sp, #-160]!
//CHECK-NEXT:        0xe4                ; end
//CHECK-NEXT:      ]
//CHECK-NEXT:      EpilogueScopes [
//CHECK-NEXT:        EpilogueScope {
//CHECK-NEXT:          StartOffset: 12
//CHECK-NEXT:          EpilogueStartIndex: 10
//CHECK-NEXT:          Opcodes [
//CHECK-NEXT:            0x83                ; ldp x29, x30, [sp], #32
//CHECK-NEXT:            0xe74e88            ; ldp q14, q15, [sp, #128]
//CHECK-NEXT:            0xe74c86            ; ldp q12, q13, [sp, #96]
//CHECK-NEXT:            0xe74a84            ; ldp q10, q11, [sp, #64]
//CHECK-NEXT:            0xe74882            ; ldp q8, q9, [sp, #32]
//CHECK-NEXT:            0xe76689            ; ldp q6, q7, [sp], #160
//CHECK-NEXT:            0xe3                ; nop
//CHECK-NEXT:            0xe3                ; nop
//CHECK-NEXT:            0xe4                ; end
//CHECK-NEXT:          ]
//CHECK-NEXT:        }
//CHECK-NEXT:      ]

//CHECK:           Prologue [
//CHECK-NEXT:        0xe70001            ; str x0, [sp, #8]
//CHECK-NEXT:        0xe70041            ; str d0, [sp, #8]
//CHECK-NEXT:        0xe70081            ; str q0, [sp, #16]
//CHECK-NEXT:        0xe72001            ; str x0, [sp, #-32]!
//CHECK-NEXT:        0xe77d01            ; stp x29, x30, [sp, #-32]!
//CHECK-NEXT:        0xe4                ; end
//CHECK-NEXT:      ]
//CHECK-NEXT:      EpilogueScopes [
//CHECK-NEXT:      ]

.section .pdata,"dr"
        .long func@IMGREL
        .long "$unwind$func"@IMGREL
        .long func2@IMGREL
        .long "$unwind$func2"@IMGREL

        .text
        .globl  func
func:
        stp q6, q7, [sp, #-160]!
        stp q8, q9, [sp, #32]
        stp q10, q11, [sp, #64]
        stp q12, q13, [sp, #96]
        stp q14, q15, [sp, #128]
        stp x29, x30, [sp, #-32]!
        mov x29, sp
        str x0, [sp, #16]
        str x9, [sp, #24]
        ldr x0, [sp, #16]
        ldr x8, [sp, #24]
        blr x8
        ldp x29, x30, [sp], #32
        ldp q14, q15, [sp, #128]
        ldp q12, q13, [sp, #96]
        ldp q10, q11, [sp, #64]
        ldp q8, q9, [sp, #32]
        ldp q6, q7, [sp], #160
        nop
        ldr x16, [x16]
        br  x16

func2:
        ret

.section .xdata,"dr"
"$unwind$func":
.byte 0x15, 0x00, 0x40, 0x40
.byte 0x0c, 0x00, 0x80, 0x02
.byte 0xe1, 0x83, 0xe6, 0xe6
.byte 0xe6, 0xe6, 0xe7, 0x66
.byte 0x89, 0xe4, 0x83, 0xe7
.byte 0x4e, 0x88, 0xe7, 0x4c
.byte 0x86, 0xe7, 0x4a, 0x84
.byte 0xe7, 0x48, 0x82, 0xe7
.byte 0x66, 0x89, 0xe3, 0xe3
.byte 0xe4, 0xe3, 0xe3, 0xe3
"$unwind$func2":
.byte 0x15, 0x00, 0x00, 0x40
.byte 0xe7, 0x00, 0x01
.byte 0xe7, 0x00, 0x41
.byte 0xe7, 0x00, 0x81
.byte 0xe7, 0x20, 0x01
.byte 0xe7, 0x7d, 0x01
.byte 0xe4
.fill 20, 1, 0xe3
