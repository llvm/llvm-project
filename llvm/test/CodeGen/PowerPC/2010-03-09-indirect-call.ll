; RUN: llc < %s -march=ppc32 -mcpu=g5 -mtriple=powerpc-apple-darwin10.0 | FileCheck %s
; ModuleID = 'nn.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-apple-darwin11.0"
; Indirect calls must use R12 on Darwin (i.e., R12 must contain the address of
; the function being called; the mtctr is not required to use it).

@p = external global void (...)*                  ; <void (...)**> [#uses=1]

define void @foo() nounwind ssp {
entry:
; CHECK: mtctr r12
; CHECK: bctrl
  %0 = load void (...)** @p, align 4              ; <void (...)*> [#uses=1]
  call void (...)* %0() nounwind
  br label %return

return:                                           ; preds = %entry
  ret void
}
