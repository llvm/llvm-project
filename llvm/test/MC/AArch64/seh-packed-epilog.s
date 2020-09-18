// This test checks that the epilogue is packed where possible.

// RUN: llvm-mc -triple aarch64-pc-win32 -filetype=obj %s -o %t.o
// RUN: llvm-readobj -u %t.o | FileCheck %s

// CHECK:      UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func
// CHECK-NEXT:     ExceptionRecord: .xdata
// CHECK-NEXT:     ExceptionData {
// CHECK-NEXT:       FunctionLength:
// CHECK-NEXT:       Version:
// CHECK-NEXT:       ExceptionData:
// CHECK-NEXT:       EpiloguePacked: Yes
// CHECK-NEXT:       EpilogueOffset: 2
// CHECK-NEXT:       ByteCodeLength:
// CHECK-NEXT:       Prologue [
// CHECK-NEXT:         0xdc04              ; str d8, [sp, #32]
// CHECK-NEXT:         0xe1                ; mov fp, sp
// CHECK-NEXT:         0x42                ; stp x29, x30, [sp, #16]
// CHECK-NEXT:         0x85                ; stp x29, x30, [sp, #-48]!
// CHECK-NEXT:         0xe6                ; save next
// CHECK-NEXT:         0x24                ; stp x19, x20, [sp, #-32]!
// CHECK-NEXT:         0xc842              ; stp x20, x21, [sp, #16]
// CHECK-NEXT:         0x03                ; sub sp, #48
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:       Epilogue [
// CHECK-NEXT:         0xe1                ; mov sp, fp
// CHECK-NEXT:         0x42                ; ldp x29, x30, [sp, #16]
// CHECK-NEXT:         0x85                ; ldp x29, x30, [sp], #48
// CHECK-NEXT:         0xe6                ; restore next
// CHECK-NEXT:         0x24                ; ldp x19, x20, [sp], #32
// CHECK-NEXT:         0xc842              ; ldp x20, x21, [sp, #16]
// CHECK-NEXT:         0x03                ; add sp, #48
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: packed2
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            ExceptionData:
// CHECK-NEXT:       EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked1
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            ExceptionData:
// CHECK-NEXT:       EpiloguePacked: No
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked2
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            ExceptionData:
// CHECK-NEXT:       EpiloguePacked: No
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked3
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            ExceptionData:
// CHECK-NEXT:       EpiloguePacked: No

    .text
    .globl func
    .seh_proc func
func:
    sub sp, sp, #48
    .seh_stackalloc 48
    // Check that canonical opcode forms (r19r20_x, fplr, fplr_x, save_next,
    // set_fp) are treated as a match even if one (in prologue or epilogue)
    // was simplified from the more generic opcodes.
    stp x20, x21, [sp, #16]
    .seh_save_regp x20, 16
    stp x19, x20, [sp, #-32]!
    .seh_save_r19r20_x 32
    stp x21, x22, [sp, #16]
    .seh_save_regp x21, 16
    stp x29, x30, [sp, #-48]!
    .seh_save_regp_x x29, 48
    stp x29, x30, [sp, #16]
    .seh_save_regp x29, 16
    add x29, sp, #0
    .seh_add_fp 0
    str d8, [sp, #32]
    .seh_save_freg d8, 32
    .seh_endprologue

    nop

    .seh_startepilogue
    mov sp, x29
    .seh_set_fp
    ldp x29, x30, [sp, #16]
    .seh_save_fplr 16
    ldp x29, x30, [sp, #-48]!
    .seh_save_fplr_x 48
    ldp x21, x22, [sp, #16]
    .seh_save_next
    ldp x19, x20, [sp], #32
    .seh_save_regp_x x19, 32
    ldp x20, x21, [sp, #16]
    .seh_save_regp x20, 16
    add sp, sp, #48
    .seh_stackalloc 48
    .seh_endepilogue
    ret
    .seh_endproc


    // Test a perfectly matching epilog with no offset.
    .seh_proc packed2
packed2:
    sub sp, sp, #48
    .seh_stackalloc 48
    stp x29, lr, [sp, #-32]!
    .seh_save_fplr_x 32
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp x29, lr, [sp], #32
    .seh_save_fplr_x 32
    add sp, sp, #48
    .seh_stackalloc 48
    .seh_endepilogue
    ret
    .seh_endproc


    .seh_proc nonpacked1
nonpacked1:
    sub sp, sp, #48
    .seh_stackalloc 48
    .seh_endprologue

    nop
    .seh_startepilogue
    add sp, sp, #48
    .seh_stackalloc 48
    .seh_endepilogue
    // This epilogue isn't packed with the prologue, as it doesn't align with
    // the end of the function (one extra nop before the ret).
    nop
    ret
    .seh_endproc


    .seh_proc nonpacked2
nonpacked2:
    sub sp, sp, #48
    .seh_stackalloc 48
    sub sp, sp, #32
    .seh_stackalloc 32
    .seh_endprologue

    nop
    .seh_startepilogue
    // Not packed; the epilogue mismatches at the second opcode.
    add sp, sp, #16
    .seh_stackalloc 16
    add sp, sp, #48
    .seh_stackalloc 48
    .seh_endepilogue
    ret
    .seh_endproc

    .seh_proc nonpacked3
nonpacked3:
    sub sp, sp, #48
    .seh_stackalloc 48
    sub sp, sp, #32
    .seh_stackalloc 32
    .seh_endprologue

    nop
    .seh_startepilogue
    // Not packed; the epilogue is longer than the prologue.
    mov sp, x29
    .seh_set_fp
    add sp, sp, #32
    .seh_stackalloc 32
    add sp, sp, #48
    .seh_stackalloc 48
    .seh_endepilogue
    ret
    .seh_endproc
