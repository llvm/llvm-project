; RUN: opt < %s -S -ipsccp | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @fn2() {
entry:
  %conv = sext i32 undef to i64
  %div = sdiv i64 8, %conv
  %call2 = call i64 @fn1(i64 %div)
  ret i64 %call2
}

; CHECK-DAG: define i64 @fn2(
; CHECK: %[[CALL:.*]] = call i64 @fn1(i64 undef)

define internal i64 @fn1(i64 %p1) {
entry:
  %tobool = icmp ne i64 %p1, 0
  %cond = select i1 %tobool, i64 %p1, i64 %p1
  ret i64 %cond
}

; CHECK-DAG: define internal i64 @fn1(
; CHECK: %[[SEL:.*]] = select i1 undef, i64 undef, i64 undef
; CHECK: ret i64 %[[SEL]]
