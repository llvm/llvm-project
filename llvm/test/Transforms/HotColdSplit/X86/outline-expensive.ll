; The magic number 6 comes from (1 * TCC_Expensive) + (1 * CostOfCallX86).
; RUN: opt -hotcoldsplit -min-outlining-thresh=6 -S < %s | FileCheck %s

; Test that we outline even though there are only two cold instructions. TTI
; should determine that they are expensive in terms of code size.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: @fun
; CHECK: call void @fun.cold.1
define void @fun(i32 %x) {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ret void

if.else:
  %y = sdiv i32 %x, 111
  call void @sink(i32 %y)
  ret void
}

declare void @sink(i32 %x) cold
