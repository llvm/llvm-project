; RUN: opt < %s -correlated-propagation -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

define void @h(i32* nocapture %p, i32 %x) local_unnamed_addr #0 {
entry:
; CHECK-LABEL: @h(
; CHECK: urem

  %cmp = icmp sgt i32 %x, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %rem2 = srem i32 %x, 10
  store i32 %rem2, i32* %p, align 4
  br label %if.end

if.end:
  ret void
}
