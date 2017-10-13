; RUN: opt < %s -correlated-propagation -S | FileCheck %s

; CHECK-LABEL: @test0(
define void @test0(i32 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %j.0 = phi i32 [ %n, %entry ], [ %div, %for.body ]
  %cmp = icmp sgt i32 %j.0, 1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
; CHECK: %div1 = udiv i32 %j.0, 2
  %div = sdiv i32 %j.0, 2
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK-LABEL: @test1(
define void @test1(i32 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %j.0 = phi i32 [ %n, %entry ], [ %div, %for.body ]
  %cmp = icmp sgt i32 %j.0, -2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
; CHECK: %div = sdiv i32 %j.0, 2
  %div = sdiv i32 %j.0, 2
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK-LABEL: @test2(
define void @test2(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 1
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: %div1 = udiv i32 %n, 2 
  %div = sdiv i32 %n, 2
  br label %exit

exit:
  ret void
}

; looping case where loop has exactly one block
; at the point of sdiv, we know that %a is always greater than 0,
; because of the guard before it, so we can transform it to udiv.
declare void @llvm.experimental.guard(i1,...)
; CHECK-LABEL: @test4
define void @test4(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
; CHECK: udiv i32 %a, 6
  %a = phi i32 [ %n, %entry ], [ %div, %loop ]
  %cond = icmp sgt i32 %a, 4
  call void(i1,...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  %div = sdiv i32 %a, 6
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

; same test as above with assume instead of guard.
declare void @llvm.assume(i1)
; CHECK-LABEL: @test5
define void @test5(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
; CHECK: udiv i32 %a, 6
  %a = phi i32 [ %n, %entry ], [ %div, %loop ]
  %cond = icmp sgt i32 %a, 4
  call void @llvm.assume(i1 %cond)
  %div = sdiv i32 %a, 6
  %loopcond = icmp sgt i32 %div, 8
  br i1 %loopcond, label %loop, label %exit

exit:
  ret void
}
