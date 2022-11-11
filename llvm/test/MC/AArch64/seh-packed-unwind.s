// Check that we generate the packed unwind info format whe possible.

// For tests that don't generate packed unwind info, we still check that
// the epilog was packed (so that the testcase otherwise had all other
// preconditions for possibly making packed unwind info).

// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o %t.o
// RUN: llvm-readobj --unwind %t.o | FileCheck %s

// CHECK:      UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func1
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 88
// CHECK-NEXT:     RegF: 7
// CHECK-NEXT:     RegI: 10
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 160
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       stp d14, d15, [sp, #128]
// CHECK-NEXT:       stp d12, d13, [sp, #112]
// CHECK-NEXT:       stp d10, d11, [sp, #96]
// CHECK-NEXT:       stp d8, d9, [sp, #80]
// CHECK-NEXT:       stp x27, x28, [sp, #64]
// CHECK-NEXT:       stp x25, x26, [sp, #48]
// CHECK-NEXT:       stp x23, x24, [sp, #32]
// CHECK-NEXT:       stp x21, x22, [sp, #16]
// CHECK-NEXT:       stp x19, x20, [sp, #-144]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func2
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 48
// CHECK-NEXT:     RegF: 2
// CHECK-NEXT:     RegI: 3
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       str d10, [sp, #40]
// CHECK-NEXT:       stp d8, d9, [sp, #24]
// CHECK-NEXT:       str x21, [sp, #16]
// CHECK-NEXT:       stp x19, x20, [sp, #-48]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func3
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 32
// CHECK-NEXT:     RegF: 3
// CHECK-NEXT:     RegI: 1
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       stp d10, d11, [sp, #24]
// CHECK-NEXT:       stp d8, d9, [sp, #8]
// CHECK-NEXT:       str x19, [sp, #-48]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func4
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 24
// CHECK-NEXT:     RegF: 1
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #32
// CHECK-NEXT:       stp d8, d9, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: notpacked_func5
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: func6
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 24
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 1
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       str lr, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func7
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 24
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 1
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       str lr, [sp, #16]
// CHECK-NEXT:       stp x19, x20, [sp, #-32]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func8
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 32
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 3
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 1
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       stp x21, lr, [sp, #16]
// CHECK-NEXT:       stp x19, x20, [sp, #-32]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func9
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 32
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 3
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #-32]!
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func10
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 24
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 3
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #-32]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func11
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 40
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 3
// CHECK-NEXT:     FrameSize: 544
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #0]
// CHECK-NEXT:       sub sp, sp, #528
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func12
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 48
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 3
// CHECK-NEXT:     FrameSize: 4112
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #0]
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       sub sp, sp, #4080
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func13
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 32
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 0
// CHECK-NEXT:     FrameSize: 4112
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       sub sp, sp, #4080
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func14
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 32
// CHECK-NEXT:     RegF: 2
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 1
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       str d10, [sp, #24]
// CHECK-NEXT:       stp d8, d9, [sp, #8]
// CHECK-NEXT:       str lr, [sp, #-32]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func15
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 20
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 3
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #-32]!
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func16
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 28
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 0
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 2
// CHECK-NEXT:     FrameSize: 32
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #-32]!
// CHECK-NEXT:       pacibsp
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func17
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 40
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 2
// CHECK-NEXT:     FrameSize: 48
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #-32]!
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       pacibsp
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func18
// CHECK-NEXT:     Fragment: No
// CHECK-NEXT:     FunctionLength: 56
// CHECK-NEXT:     RegF: 0
// CHECK-NEXT:     RegI: 2
// CHECK-NEXT:     HomedParameters: No
// CHECK-NEXT:     CR: 2
// CHECK-NEXT:     FrameSize: 4112
// CHECK-NEXT:     Prologue [
// CHECK-NEXT:       mov x29, sp
// CHECK-NEXT:       stp x29, lr, [sp, #0]
// CHECK-NEXT:       sub sp, sp, #16
// CHECK-NEXT:       sub sp, sp, #4080
// CHECK-NEXT:       stp x19, x20, [sp, #-16]!
// CHECK-NEXT:       pacibsp
// CHECK-NEXT:       end
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked1
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked2
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked3
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked4
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked5
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked6
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked7
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked8
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked9
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked10
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked11
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked12
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked13
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked14
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes
// CHECK:        RuntimeFunction {
// CHECK-NEXT:     Function: nonpacked15
// CHECK-NEXT:     ExceptionRecord:
// CHECK-NEXT:     ExceptionData {
// CHECK:            EpiloguePacked: Yes

    .text
func1:
    .seh_proc func1
    stp x19, x20, [sp, #-144]!
    .seh_save_r19r20_x 144
    stp x21, x22, [sp, #16]
    .seh_save_regp x21, 16
    stp x23, x24, [sp, #32]
    .seh_save_next
    stp x25, x26, [sp, #48]
    .seh_save_next
    stp x27, x28, [sp, #64]
    .seh_save_next
    stp d8,  d9,  [sp, #80]
    .seh_save_fregp d8, 80
    stp d10, d11, [sp, #96]
    .seh_save_fregp d10, 96
    stp d12, d13, [sp, #112]
    .seh_save_fregp d12, 112
    stp d14, d15, [sp, #128]
    .seh_save_fregp d14, 128
    sub sp,  sp,  #16
    .seh_stackalloc 16
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp,  sp,  #16
    .seh_stackalloc 16
    ldp d14, d15, [sp, #128]
    .seh_save_fregp d14, 128
    ldp d12, d13, [sp, #112]
    .seh_save_fregp d12, 112
    ldp d10, d11, [sp, #96]
    .seh_save_fregp d10, 96
    ldp d8,  d9,  [sp, #80]
    .seh_save_fregp d8, 80
    ldp x27, x28, [sp, #64]
    .seh_save_next
    ldp x25, x26, [sp, #48]
    .seh_save_next
    ldp x23, x24, [sp, #32]
    .seh_save_next
    ldp x21, x22, [sp, #16]
    .seh_save_next
    ldp x19, x20, [sp], #144
    .seh_save_regp_x x19, 144
    .seh_endepilogue
    ret
    .seh_endproc

func2:
    .seh_proc func2
    stp x19, x20, [sp, #-48]!
    .seh_save_r19r20_x 48
    str x21,      [sp, #16]
    .seh_save_reg x21, 16
    stp d8,  d9,  [sp, #24]
    .seh_save_fregp d8, 24
    str d10,      [sp, #40]
    .seh_save_freg d10, 40
    sub sp,  sp,  #0
    .seh_stackalloc 0
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp,  sp,  #0
    .seh_stackalloc 0
    ldr d10,      [sp, #40]
    .seh_save_freg d10, 40
    ldp d8,  d9,  [sp, #24]
    .seh_save_fregp d8, 24
    ldr x21,      [sp, #16]
    .seh_save_reg x21, 16
    ldp x19, x20, [sp], #48
    .seh_save_r19r20_x 48
    .seh_endepilogue
    ret
    .seh_endproc

func3:
    .seh_proc func3
    str x19,      [sp, #-48]!
    .seh_save_reg_x x19, 48
    stp d8,  d9,  [sp, #8]
    .seh_save_fregp d8, 8
    stp d10, d11, [sp, #24]
    .seh_save_fregp d10, 24
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp d10, d11, [sp, #24]
    .seh_save_fregp d10, 24
    ldp d8,  d9,  [sp, #8]
    .seh_save_fregp d8, 8
    ldr x19,      [sp], #48
    .seh_save_reg_x x19, 48
    .seh_endepilogue
    ret
    .seh_endproc

func4:
    .seh_proc func4
    stp d8,  d9,  [sp, #-16]!
    .seh_save_fregp_x d8, 16
    sub sp,  sp,  #32
    .seh_stackalloc 32
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp,  sp,  #32
    .seh_stackalloc 32
    ldp d8,  d9,  [sp], #16
    .seh_save_fregp_x d8, 16
    .seh_endepilogue
    ret
    .seh_endproc

notpacked_func5:
    .seh_proc notpacked_func5
    str x19, [sp, #-80]!
    .seh_save_reg_x x19, 80
    stp x0,  x1,  [sp, #16]
    .seh_nop
    stp x2,  x3,  [sp, #32]
    .seh_nop
    stp x4,  x5,  [sp, #48]
    .seh_nop
    stp x6,  x7,  [sp, #64]
    .seh_nop
    sub sp,  sp,  #32
    .seh_stackalloc 32
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp,  sp,  #32
    .seh_stackalloc 32
    nop
    .seh_nop
    nop
    .seh_nop
    nop
    .seh_nop
    nop
    .seh_nop
    ldr x19, [sp], #80
    .seh_save_reg_x x19, 80
    .seh_endepilogue
    ret
    .seh_endproc

func6:
    .seh_proc func6
    str lr,       [sp, #-16]!
    .seh_save_reg_x lr, 16
    sub sp,  sp,  #16
    .seh_stackalloc 16
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp,  sp,  #16
    .seh_stackalloc 16
    ldr lr,       [sp], #16
    .seh_save_reg_x lr, 16
    .seh_endepilogue
    ret
    .seh_endproc

func7:
    .seh_proc func7
    stp x19, x20, [sp, #-32]!
    .seh_save_r19r20_x 32
    str lr,       [sp, #16]
    .seh_save_reg lr, 16
    .seh_endprologue
    nop
    .seh_startepilogue
    ldr lr,       [sp, #16]
    .seh_save_reg lr, 16
    ldp x19, x20, [sp], #32
    .seh_save_r19r20_x 32
    .seh_endepilogue
    ret
    .seh_endproc

func8:
    .seh_proc func8
    stp x19, x20, [sp, #-32]!
    .seh_save_r19r20_x 32
    stp x21, lr,  [sp, #16]
    .seh_save_lrpair x21, 16
    sub sp,  sp,  #16
    .seh_stackalloc 16
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp,  sp,  #16
    .seh_stackalloc 16
    ldp x21, lr,  [sp, #16]
    .seh_save_lrpair x21, 16
    ldp x19, x20, [sp], #32
    .seh_save_r19r20_x 32
    .seh_endepilogue
    ret
    .seh_endproc

func9:
    .seh_proc func9
    stp x19, x20, [sp, #-16]!
    .seh_save_r19r20_x 16
    stp x29, lr,  [sp, #-32]!
    .seh_save_fplr_x 32
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    mov sp,  x29
    .seh_set_fp
    ldp x29, lr,  [sp], #32
    .seh_save_fplr_x 32
    ldp x19, x20, [sp], #16
    .seh_save_r19r20_x 16
    .seh_endepilogue
    ret
    .seh_endproc

func10:
    .seh_proc func10
    stp x29, lr,  [sp, #-32]!
    .seh_save_fplr_x 32
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    mov sp,  x29
    .seh_set_fp
    ldp x29, lr,  [sp], #32
    .seh_save_fplr_x 32
    .seh_endepilogue
    ret
    .seh_endproc

func11:
    .seh_proc func11
    stp x19, x20, [sp, #-16]!
    .seh_save_r19r20_x 16
    sub sp,  sp,  #528
    .seh_stackalloc 528
    stp x29, lr,  [sp, #0]
    .seh_save_fplr 0
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    mov sp,  x29
    .seh_set_fp
    ldp x29, lr,  [sp, #0]
    .seh_save_fplr 0
    add sp,  sp,  #528
    .seh_stackalloc 528
    ldp x19, x20, [sp], #16
    .seh_save_r19r20_x 16
    .seh_endepilogue
    ret
    .seh_endproc

func12:
    .seh_proc func12
    stp x19, x20, [sp, #-16]!
    .seh_save_r19r20_x 16
    sub sp,  sp,  #4080
    .seh_stackalloc 4080
    sub sp,  sp,  #16
    .seh_stackalloc 16
    stp x29, lr,  [sp, #0]
    .seh_save_fplr 0
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    mov sp,  x29
    .seh_set_fp
    ldp x29, lr,  [sp, #0]
    .seh_save_fplr 0
    add sp,  sp,  #16
    .seh_stackalloc 16
    add sp,  sp,  #4080
    .seh_stackalloc 4080
    ldp x19, x20, [sp], #16
    .seh_save_r19r20_x 16
    .seh_endepilogue
    ret
    .seh_endproc

func13:
    .seh_proc func13
    stp x19, x20, [sp, #-16]!
    .seh_save_r19r20_x 16
    sub sp,  sp,  #4080
    .seh_stackalloc 4080
    sub sp,  sp,  #16
    .seh_stackalloc 16
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp,  sp,  #16
    .seh_stackalloc 16
    add sp,  sp,  #4080
    .seh_stackalloc 4080
    ldp x19, x20, [sp], #16
    .seh_save_r19r20_x 16
    .seh_endepilogue
    ret
    .seh_endproc

func14:
    .seh_proc func14
    str lr,       [sp, #-32]!
    .seh_save_reg_x lr, 32
    stp d8,  d9,  [sp, #8]
    .seh_save_fregp d8, 8
    str d10,      [sp, #24]
    .seh_save_freg d10, 24
    .seh_endprologue
    nop
    .seh_startepilogue
    ldr d10,      [sp, #24]
    .seh_save_freg d10, 24
    ldp d8,  d9,  [sp, #8]
    .seh_save_fregp d8, 8
    ldr lr,       [sp], #32
    .seh_save_reg_x lr, 32
    .seh_endepilogue
    ret
    .seh_endproc

func15:
    .seh_proc func15
    stp x29, lr,  [sp, #-32]!
    .seh_save_fplr_x 32
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    // Epilogue missing the .seh_set_fp, but still generating packed info.
    ldp x29, lr,  [sp], #32
    .seh_save_fplr_x 32
    .seh_endepilogue
    ret
    .seh_endproc

func16:
    .seh_proc func16
    pacibsp
    .seh_pac_sign_lr
    stp x29, lr,  [sp, #-32]!
    .seh_save_fplr_x 32
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp x29, lr, [sp], #32
    .seh_save_fplr_x 32
    autibsp
    .seh_pac_sign_lr
    .seh_endepilogue
    ret
    .seh_endproc

func17:
    .seh_proc func17
    pacibsp
    .seh_pac_sign_lr
    stp x19, x20, [sp, #-16]!
    .seh_save_r19r20_x 16
    stp x29, lr,  [sp, #-32]!
    .seh_save_fplr_x 32
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    mov sp,  x29
    .seh_set_fp
    ldp x29, lr,  [sp], #32
    .seh_save_fplr_x 32
    ldp x19, x20, [sp], #16
    .seh_save_r19r20_x 16
    autibsp
    .seh_pac_sign_lr
    .seh_endepilogue
    ret
    .seh_endproc

func18:
    .seh_proc func18
    pacibsp
    .seh_pac_sign_lr
    stp x19, x20, [sp, #-16]!
    .seh_save_r19r20_x 16
    sub sp,  sp,  #4080
    .seh_stackalloc 4080
    sub sp,  sp,  #16
    .seh_stackalloc 16
    stp x29, lr,  [sp, #0]
    .seh_save_fplr 0
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    mov sp,  x29
    .seh_set_fp
    ldp x29, lr,  [sp, #0]
    .seh_save_fplr 0
    add sp,  sp,  #16
    .seh_stackalloc 16
    add sp,  sp,  #4080
    .seh_stackalloc 4080
    ldp x19, x20, [sp], #16
    .seh_save_r19r20_x 16
    autibsp
    .seh_pac_sign_lr
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked1:
    .seh_proc nonpacked1
    // Can't be packed; can't save integer registers after float registers.
    stp d8,  d9,  [sp, #-32]!
    .seh_save_fregp_x d8, 32
    stp x19, x20, [sp, #16]!
    .seh_save_regp x19, 16
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp x19, x20, [sp, #16]
    .seh_save_regp x19, 16
    ldp d8,  d9,  [sp], #32
    .seh_save_fregp_x d8, 32
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked2:
    .seh_proc nonpacked2
    // Can't be packed; x21/x22 aren't saved in the expected spot
    stp x19, x20, [sp, #-48]!
    .seh_save_r19r20_x 48
    stp x21, x22, [sp, #32]
    .seh_save_regp x21, 32
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp x21, x22, [sp, #32]
    .seh_save_regp x21, 32
    ldp x19, x20, [sp], #48
    .seh_save_r19r20_x 48
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked3:
    .seh_proc nonpacked3
    // Can't be packed; x29/x30 can't be treated as the other saved registers
    stp x19, x20, [sp, #-96]!
    .seh_save_r19r20_x 96
    stp x21, x22, [sp, #16]
    .seh_save_next
    stp x23, x24, [sp, #32]
    .seh_save_next
    stp x25, x26, [sp, #48]
    .seh_save_next
    stp x27, x28, [sp, #64]
    .seh_save_next
    stp x29, x30, [sp, #80]
    .seh_save_next
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp x29, x30, [sp, #80]
    .seh_save_next
    ldp x27, x28, [sp, #64]
    .seh_save_next
    ldp x25, x26, [sp, #48]
    .seh_save_next
    ldp x23, x24, [sp, #32]
    .seh_save_next
    ldp x21, x22, [sp, #16]
    .seh_save_next
    ldp x19, x20, [sp], #96
    .seh_save_r19r20_x 96
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked4:
    .seh_proc nonpacked4
    // Can't be packed; more predecrement for x19/x20 than used for
    // corresponding RegI/RegF/LR saves
    stp x19, x20, [sp, #-32]!
    .seh_save_r19r20_x 32
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp x19, x20, [sp], #32
    .seh_save_r19r20_x 32
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked5:
    .seh_proc nonpacked5
    // Can't be packed; can't save LR twice
    stp x19, x20, [sp, #-32]!
    .seh_save_r19r20_x 32
    str lr, [sp, #16]
    .seh_save_reg lr, 16
    str lr, [sp, #24]
    .seh_save_reg lr, 24
    .seh_endprologue
    nop
    .seh_startepilogue
    ldr lr, [sp, #24]
    .seh_save_reg lr, 24
    ldr lr, [sp, #16]
    .seh_save_reg lr, 16
    ldp x19, x20, [sp], #32
    .seh_save_r19r20_x 32
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked6:
    .seh_proc nonpacked6
    // Can't be packed; can't save LR both standalone (CR 1) and as FPLR (CR 3)
    stp x19, x20, [sp, #-32]!
    .seh_save_r19r20_x 32
    str lr, [sp, #16]
    .seh_save_reg lr, 16
    stp x29, lr,  [sp, #-16]!
    .seh_save_fplr_x 16
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    mov sp,  x29
    .seh_set_fp
    ldp x29, lr,  [sp], #32
    .seh_save_fplr_x 16
    ldr lr, [sp, #16]
    .seh_save_reg lr, 16
    ldp x19, x20, [sp], #32
    .seh_save_r19r20_x 32
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked7:
    .seh_proc nonpacked7
    // Can't be packed; too many saved FP regs
    stp d8,  d9,  [sp, #-80]!
    .seh_save_fregp_x d8, 80
    stp d10, d11, [sp, #16]
    .seh_save_fregp d10, 16
    stp d12, d13, [sp, #32]
    .seh_save_fregp d12, 32
    stp d14, d15, [sp, #48]
    .seh_save_fregp d14, 48
    stp d16, d17, [sp, #64]
    .seh_save_next
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp d16, d17, [sp, #64]
    .seh_save_next
    ldp d14, d15, [sp, #48]
    .seh_save_fregp d14, 48
    ldp d12, d13, [sp, #32]
    .seh_save_fregp d12, 32
    ldp d10, d11, [sp, #16]
    .seh_save_fregp d10, 16
    ldp d8,  d9,  [sp], #80
    .seh_save_fregp_x d8, 80
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked8:
    .seh_proc nonpacked8
    // Can't be packed; Can't handle only a single FP reg
    str d8,  [sp, #-16]!
    .seh_save_freg_x d8, 16
    .seh_endprologue
    nop
    .seh_startepilogue
    ldr d8,  [sp], #16
    .seh_save_freg_x d8, 16
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked9:
    .seh_proc nonpacked9
    // Can't be packed; can't have a separate stack adjustment with save_fplr_x
    sub sp, sp, #32
    .seh_stackalloc 32
    stp x29, lr,  [sp, #-16]!
    .seh_save_fplr_x 16
    mov x29, sp
    .seh_set_fp
    .seh_endprologue
    nop
    .seh_startepilogue
    mov sp,  x29
    .seh_set_fp
    ldp x29, lr,  [sp], #32
    .seh_save_fplr_x 16
    add sp, sp, #32
    .seh_stackalloc 32
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked10:
    .seh_proc nonpacked10
    // Can't be packed; wrong predecrement
    stp x19, x20, [sp, #-16]!
    .seh_save_r19r20_x 16
    stp x21, x22, [sp, #16]
    .seh_save_next
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp x21, x22, [sp, #16]
    .seh_save_next
    ldp x19, x20, [sp], #16
    .seh_save_r19r20_x 16
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked11:
    .seh_proc nonpacked11
    // Can't be packed; too big stack allocation
    sub sp, sp, #4080
    .seh_stackalloc 4080
    sub sp, sp, #8192
    .seh_stackalloc 8192
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp, sp, #8192
    .seh_stackalloc 8192
    add sp, sp, #4080
    .seh_stackalloc 4080
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked12:
    .seh_proc nonpacked12
    // Can't be packed; missing .seh_set_fp
    stp x29, lr,  [sp, #-32]!
    .seh_save_fplr_x 32
    .seh_endprologue
    nop
    .seh_startepilogue
    ldp x29, lr,  [sp], #32
    .seh_save_fplr_x 32
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked13:
    .seh_proc nonpacked13
    // Can't be packed; not doing a packed info if .seh_handlerdata is used
    sub sp, sp, #16
    .seh_stackalloc 16
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp, sp, #16
    .seh_stackalloc 16
    .seh_endepilogue
    ret
    .seh_endfunclet
    .seh_handlerdata
    .long 0
    .text
    .seh_endproc

nonpacked14:
    .seh_proc nonpacked14
    // Can't be packed; a signed return address can only be expressed if
    // we save both x29 and lr on the stack.
    pacibsp
    .seh_pac_sign_lr
    str lr,       [sp, #-32]!
    .seh_save_reg_x lr, 32
    .seh_endprologue
    nop
    .seh_startepilogue
    ldr lr,       [sp], #32
    .seh_save_reg_x lr, 32
    autibsp
    .seh_pac_sign_lr
    .seh_endepilogue
    ret
    .seh_endproc

nonpacked15:
    .seh_proc nonpacked15
    // Can't be packed; a signed return address can only be expressed if
    // we save both x29 and lr on the stack.
    pacibsp
    .seh_pac_sign_lr
    stp x19, x20, [sp, #-32]!
    .seh_save_r19r20_x 32
    stp x21, lr,  [sp, #16]
    .seh_save_lrpair x21, 16
    sub sp,  sp,  #16
    .seh_stackalloc 16
    .seh_endprologue
    nop
    .seh_startepilogue
    add sp,  sp,  #16
    .seh_stackalloc 16
    ldp x21, lr,  [sp, #16]
    .seh_save_lrpair x21, 16
    ldp x19, x20, [sp], #32
    .seh_save_r19r20_x 32
    autibsp
    .seh_pac_sign_lr
    .seh_endepilogue
    ret
    .seh_endproc
