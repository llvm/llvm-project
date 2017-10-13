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

; looping case where loop has exactly one block
; at the point of srem, we know that %a is always greater than 0,
; because of the assume before it, so we can transform it to urem.
declare void @llvm.assume(i1)
; CHECK-LABEL: @test4
define void @test4(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
; CHECK: urem i32 %a, 6
  %a = phi i32 [ %n, %entry ], [ %rem, %loop ]
  %cond = icmp sgt i32 %a, 4
  call void @llvm.assume(i1 %cond)
  %rem = srem i32 %a, 6
  %loopcond = icmp sgt i32 %rem, 8
  br i1 %loopcond, label %loop, label %exit

exit:
  ret void
}
