// This test checks that the unwinding opcodes are remapped to more
// efficient ones where possible.

// RUN: llvm-mc -triple aarch64-pc-win32 -filetype=obj %s -o %t.o
// RUN: llvm-readobj -u %t.o | FileCheck %s

// CHECK:      UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func
// CHECK-NEXT:     ExceptionRecord: .xdata
// CHECK-NEXT:     ExceptionData {
// CHECK:            Prologue [
// CHECK-NEXT:         0xd882              ; stp d10, d11, [sp, #16]
// CHECK-NEXT:         0xda07              ; stp d8, d9, [sp, #-64]!
// CHECK-NEXT:         0xe6                ; save next
// CHECK-NEXT:         0x28                ; stp x19, x20, [sp, #-64]!
// CHECK-NEXT:         0xca49              ; stp x28, x29, [sp, #72]
// CHECK-NEXT:         0xe6                ; save next
// CHECK-NEXT:         0xe6                ; save next
// CHECK-NEXT:         0xe6                ; save next
// CHECK-NEXT:         0xcc47              ; stp x20, x21, [sp, #-64]!
// CHECK-NEXT:         0x42                ; stp x29, x30, [sp, #16]
// CHECK-NEXT:         0xca02              ; stp x27, x28, [sp, #16]
// CHECK-NEXT:         0x83                ; stp x29, x30, [sp, #-32]!
// CHECK-NEXT:         0xce03              ; stp x27, x28, [sp, #-32]!
// CHECK-NEXT:         0xe1                ; mov fp, sp
// CHECK-NEXT:         0xe201              ; add fp, sp, #8
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:       Epilogue [
// CHECK-NEXT:         0xc904              ; ldp x23, x24, [sp, #32]
// CHECK-NEXT:         0xe6                ; restore next
// CHECK-NEXT:         0xcc83              ; ldp x21, x22, [sp], #32
// CHECK-NEXT:         0x24                ; ldp x19, x20, [sp], #32
// CHECK-NEXT:         0xcc1f              ; ldp x19, x20, [sp], #256
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]


    .text
    .globl func
    .seh_proc func
func:
    add x29, sp, #8
    .seh_add_fp 8
    add x29, sp, #0
    .seh_add_fp 0

    stp x27, x28, [sp, #-32]!
    .seh_save_regp_x x27, 32
    stp x29, x30, [sp, #-32]!
    .seh_save_regp_x x29, 32

    stp x27, x28, [sp, #16]
    .seh_save_regp x27, 16
    stp x29, x30, [sp, #16]
    .seh_save_regp x29, 16

    stp x20, x21, [sp, #-64]!
    .seh_save_regp_x x20, 64
    stp x22, x23, [sp, #16]
    .seh_save_regp x22, 16
    stp x24, x25, [sp, #32]
    .seh_save_next
    stp x26, x27, [sp, #48]
    .seh_save_regp x26, 48
    stp x28, x29, [sp, #72]
    .seh_save_regp x28, 72

    stp x19, x20, [sp, #-64]!
    .seh_save_r19r20_x 64
    stp x21, x22, [sp, #16]
    .seh_save_regp x21, 16

    stp d8,  d9,  [sp, #-64]!
    .seh_save_fregp_x d8, 64
    stp d10, d11, [sp, #16]
    // This is intentionally not converted into a save_next, to avoid
    // bugs in the windows unwinder.
    .seh_save_fregp d10, 16

    .seh_endprologue

    nop

    .seh_startepilogue
    ldp x27, x28, [sp, #32]
    .seh_save_regp x23, 32
    ldp x23, x24, [sp, #16]
    .seh_save_regp x23, 16
    ldp x21, x22, [sp], #32
    .seh_save_regp_x x21, 32
    ldp x19, x20, [sp], #32
    .seh_save_regp_x x19, 32
    ldp x19, x20, [sp], #256
    .seh_save_regp_x x19, 256
    .seh_endepilogue
    ret
    .seh_endproc
