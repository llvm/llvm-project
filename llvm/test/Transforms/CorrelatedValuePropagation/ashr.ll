; RUN: opt < %s -correlated-propagation -S | FileCheck %s

; CHECK-LABEL: @test1
define void @test1(i32 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %a = phi i32 [ %n, %entry ], [ %shr, %for.body ]
  %cmp = icmp sgt i32 %a, 1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
; CHECK: lshr i32 %a, 5
  %shr = ashr i32 %a, 5
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

;; Negative test to show transform doesn't happen unless n > 0.
; CHECK-LABEL: @test2
define void @test2(i32 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %a = phi i32 [ %n, %entry ], [ %shr, %for.body ]
  %cmp = icmp sgt i32 %a, -2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
; CHECK: ashr i32 %a, 2
  %shr = ashr i32 %a, 2
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

;; Non looping test case.
; CHECK-LABEL: @test3
define void @test3(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: lshr exact i32 %n, 4
  %shr = ashr exact i32 %n, 4
  br label %exit

exit:
  ret void
}
