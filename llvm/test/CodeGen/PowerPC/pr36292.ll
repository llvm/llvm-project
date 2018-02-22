; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-unknown < %s  | \
; RUN:   FileCheck %s --implicit-check-not=mtctr --implicit-check-not=bdnz
$test = comdat any

; No CTR loop due to frem (since it is always a call).
define void @test() #0 comdat {
; CHECK-LABEL: test:
; CHECK:    ld 29, 0(3)
; CHECK:    ld 30, 40(1)
; CHECK:    xxlxor 31, 31, 31
; CHECK:    cmpld 30, 29
; CHECK-NEXT:    bge- 0, .LBB0_2
; CHECK-NEXT:    .p2align 5
; CHECK-NEXT:  .LBB0_1: # %bounds.ok
; CHECK:    fmr 1, 31
; CHECK-NEXT:    lfsx 2, 0, 3
; CHECK-NEXT:    bl fmodf
; CHECK-NEXT:    nop
; CHECK-NEXT:    addi 30, 30, 1
; CHECK-NEXT:    stfsx 1, 0, 3
; CHECK-NEXT:    cmpld 30, 29
; CHECK-NEXT:    blt+ 0, .LBB0_1
; CHECK-NEXT:  .LBB0_2: # %bounds.fail
; CHECK-NEXT:    std 30, 40(1)
  %pos = alloca i64, align 8
  br label %forcond

forcond:                                          ; preds = %bounds.ok, %0
  %1 = load i64, i64* %pos
  %.len1 = load i64, i64* undef
  %bounds.cmp = icmp ult i64 %1, %.len1
  br i1 %bounds.cmp, label %bounds.ok, label %bounds.fail

bounds.ok:                                        ; preds = %forcond
  %2 = load float, float* undef
  %3 = frem float 0.000000e+00, %2
  store float %3, float* undef
  %4 = load i64, i64* %pos
  %5 = add i64 %4, 1
  store i64 %5, i64* %pos
  br label %forcond

bounds.fail:                                      ; preds = %forcond
  unreachable
}

