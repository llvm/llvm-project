; RUN: opt < %s -instcombine -S | FileCheck %s

;; Target triple for gep raising case below.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

define float @test1(float %A, float %B, i1 %C) {
EntryBlock:
  ;; A*(1 - uitofp i1 C) -> select C, 0, A
  %cf = uitofp i1 %C to float
  %mc = fsub float 1.000000e+00, %cf
  %p1 = fmul fast float %A, %mc
  ret float %p1
; CHECK: @test1
; CHECK: select i1 %C, float -0.000000e+00, float %A
}

define float @test2(float %A, float %B, i1 %C) {
EntryBlock:
  ;; B*(uitofp i1 C) -> select C, B, 0
  %cf = uitofp i1 %C to float
  %p2 = fmul fast float %B, %cf
  ret float %p2
; CHECK: @test2
; CHECK: select i1 %C, float %B, float -0.000000e+00
}

define float @test3(float %A, float %B, i1 %C) {
EntryBlock:
  ;; A*(1 - uitofp i1 C) + B*(uitofp i1 C) -> select C, A, B
  %cf = uitofp i1 %C to float
  %mc = fsub float 1.000000e+00, %cf
  %p1 = fmul fast float %A, %mc
  %p2 = fmul fast float %B, %cf
  %s1 = fadd fast float %p1, %p2
  ret float %s1
; CHECK: @test3
; CHECK: select i1 %C, float %B, float %A
}

; PR15952
define float @test4(float %A, float %B, i32 %C) {
  %cf = uitofp i32 %C to float
  %mc = fsub float 1.000000e+00, %cf
  %p1 = fmul fast float %A, %mc
  ret float %p1
; CHECK: @test4
; CHECK: uitofp
}

define float @test5(float %A, float %B, i32 %C) {
  %cf = uitofp i32 %C to float
  %p2 = fmul fast float %B, %cf
  ret float %p2
; CHECK: @test5
; CHECK: uitofp
}

