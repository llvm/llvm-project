; RUN: llc -mcpu=ppc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Tests that the 'nest' parameter attribute causes the relevant parameter to be
; passed in the right register (r11 for PPC).

define ptr @nest_receiver(ptr nest %arg) nounwind {
; CHECK-LABEL: nest_receiver:
; CHECK:       .p2align	3, 0x0
; CHECK-NEXT:    .quad	.Lfunc_begin0
; CHECK-NEXT:    .quad	.TOC.@tocbase
; CHECK-NEXT:    .quad	0
; CHECK-NEXT:    .text
; CHECK-NEXT:    .Lfunc_begin0:
; CHECK-NEXT:    # %bb.0:
; CHECK-NEXT:    mr	3, 11
; CHECK-NEXT:    blr
; CHECK-NEXT:    .long	0
; CHECK-NEXT:    .quad	0

  ret ptr %arg
}

define ptr @nest_caller(ptr %arg) nounwind {
; CHECK-LABEL: nest_caller:
; CHECK:       .p2align	3, 0x0
; CHECK-NEXT:    .quad	.Lfunc_begin1
; CHECK-NEXT:    .quad	.TOC.@tocbase
; CHECK-NEXT:    .quad	0
; CHECK-NEXT:    .text
; CHECK-NEXT:    .Lfunc_begin1:
; CHECK-NEXT:    # %bb.0:
; CHECK-NEXT:    mflr 0
; CHECK-NEXT:    stdu 1, -112(1)
; CHECK-NEXT:    std 0, 128(1)
; CHECK-NEXT:    mr	11, 3
; CHECK-NEXT:    bl nest_receiver
; CHECK-NEXT:    nop
; CHECK-NEXT:    addi 1, 1, 112
; CHECK-NEXT:    ld 0, 16(1)
; CHECK-NEXT:    mtlr 0
; CHECK-NEXT:    blr
; CHECK-NEXT:    .long	0
; CHECK-NEXT:    .quad	0

  %result = call ptr @nest_receiver(ptr nest %arg)
  ret ptr %result
}

define void @test_indirect(ptr nocapture %f, ptr %p) {
entry:

; CHECK-LABEL: test_indirect:
; CHECK:       .p2align	3, 0x0
; CHECK-NEXT:    .quad	.Lfunc_begin2
; CHECK-NEXT:    .quad	.TOC.@tocbase
; CHECK-NEXT:    .quad	0
; CHECK-NEXT:    .text
; CHECK-NEXT:    .Lfunc_begin2:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    # %bb.0:                                # %entry
; CHECK-NEXT:    mflr 0
; CHECK-NEXT:    stdu 1, -112(1)
; CHECK-NEXT:    std 0, 128(1)
; CHECK-NEXT:    .cfi_def_cfa_offset 112
; CHECK-NEXT:    .cfi_offset lr, 16
; CHECK-NEXT:    ld 5, 0(3)
; CHECK-NEXT:    std 2, 40(1)
; CHECK-NEXT:    ld 2, 8(3)
; CHECK-NEXT:    mr	11, 4
; CHECK-NEXT:    mtctr 5
; CHECK-NEXT:    bctrl
; CHECK-NEXT:    ld 2, 40(1)
; CHECK-NEXT:    addi 1, 1, 112
; CHECK-NEXT:    ld 0, 16(1)
; CHECK-NEXT:    mtlr 0
; CHECK-NEXT:    blr
; CHECK-NEXT:    .long	0
; CHECK-NEXT:    .quad	0

  %call = tail call signext i32 %f(ptr nest %p)
  ret void
}

