; RUN: opt < %s -passes=indvars -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

declare void @foo(i64)

; CHECK-LABEL: @test
define void @test(i64 %a) {
entry:
  br label %outer_header

outer_header:
  %i = phi i64 [20, %entry], [%i.next, %outer_latch]
  %i.next = add nuw nsw i64 %i, 1
  br label %inner_header

inner_header:
  %j = phi i64 [1, %outer_header], [%j.next, %inner_header]
  %cmp = icmp ult i64 %j, %i.next
; CHECK-NOT: select
  %s = select i1 %cmp, i64 %j, i64 %i
; CHECK: call void @foo(i64 %j)
  call void @foo(i64 %s)
  %j.next = add nuw nsw i64 %j, 1
  %cond = icmp ult i64 %j, %i
  br i1 %cond, label %inner_header, label %outer_latch

outer_latch:
  %cond2 = icmp ne i64 %i.next, 40
  br i1 %cond2, label %outer_header, label %return

return:
  ret void
}
