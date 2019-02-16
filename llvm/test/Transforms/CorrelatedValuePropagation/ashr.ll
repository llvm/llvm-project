; RUN: opt < %s -correlated-propagation -S | FileCheck %s

; Check that debug locations are preserved. For more info see:
;   https://llvm.org/docs/SourceLevelDebugging.html#fixing-errors
; RUN: opt < %s -enable-debugify -correlated-propagation -S 2>&1 | \
; RUN:   FileCheck %s -check-prefix=DEBUG
; DEBUG: CheckModuleDebugify: PASS

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

; looping case where loop has exactly one block
; at the point of ashr, we know that the operand is always greater than 0,
; because of the guard before it, so we can transform it to lshr.
declare void @llvm.experimental.guard(i1,...)
; CHECK-LABEL: @test4
define void @test4(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
; CHECK: lshr i32 %a, 1
  %a = phi i32 [ %n, %entry ], [ %shr, %loop ]
  %cond = icmp sgt i32 %a, 2
  call void(i1,...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  %shr = ashr i32 %a, 1
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
; CHECK: lshr i32 %a, 1
  %a = phi i32 [ %n, %entry ], [ %shr, %loop ]
  %cond = icmp sgt i32 %a, 4
  call void @llvm.assume(i1 %cond)
  %shr = ashr i32 %a, 1
  %loopcond = icmp sgt i32 %shr, 8
  br i1 %loopcond, label %loop, label %exit

exit:
  ret void
}
