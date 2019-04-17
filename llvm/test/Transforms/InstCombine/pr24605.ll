; RUN: opt -S -instcombine < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i1 @f(i8* %a, i8 %b) {
; CHECK-LABEL: @f(
entry:
  %or = or i8 %b, -117
  %sub = add i8 %or, -1
  store i8 %sub, i8* %a, align 1
  %cmp = icmp ugt i8 %or, %sub
  ret i1 %cmp
; CHECK: ret i1 true
}
