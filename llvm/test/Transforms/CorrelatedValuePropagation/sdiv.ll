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
