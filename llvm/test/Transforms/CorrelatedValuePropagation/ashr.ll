; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; Check that debug locations are preserved. For more info see:
;   https://llvm.org/docs/SourceLevelDebugging.html#fixing-errors
; RUN: opt < %s -enable-debugify -passes=correlated-propagation -S 2>&1 | \
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

; check that ashr of -1 or 0 is optimized away
; CHECK-LABEL: @test6
define i32 @test6(i32 %f, i32 %g) {
entry:
  %0 = add i32 %f, 1
  %1 = icmp ult i32 %0, 2
  tail call void @llvm.assume(i1 %1)
; CHECK: ret i32 %f
  %shr = ashr i32 %f, %g
  ret i32 %shr
}

; same test as above with different numbers
; CHECK-LABEL: @test7
define i32 @test7(i32 %f, i32 %g) {
entry:
  %0 = and i32 %f, -2
  %1 = icmp eq i32 %0, 6
  tail call void @llvm.assume(i1 %1)
  %sub = add nsw i32 %f, -7
; CHECK: ret i32 %sub
  %shr = ashr i32 %sub, %g
  ret i32 %shr
}

; check that ashr of -2 or 1 is not optimized away
; CHECK-LABEL: @test8
define i32 @test8(i32 %f, i32 %g, i1 %s) {
entry:
; CHECK: ashr i32 -2, %f
  %0 = ashr i32 -2, %f
; CHECK: lshr i32 1, %g
  %1 = ashr i32 1, %g
  %2 = select i1 %s, i32 %0, i32 %1
  ret i32 %2
}

define i32 @may_including_undef(i1 %c.1, i1 %c.2) {
; CHECK-LABEL: define i32 @may_including_undef
; CHECK-SAME: (i1 [[C_1:%.*]], i1 [[C_2:%.*]]) {
; CHECK-NEXT:    br i1 [[C_1]], label [[TRUE_1:%.*]], label [[FALSE:%.*]]
; CHECK:       true.1:
; CHECK-NEXT:    br i1 [[C_2]], label [[TRUE_2:%.*]], label [[EXIT:%.*]]
; CHECK:       true.2:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       false:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    [[P:%.*]] = phi i32 [ 2, [[TRUE_1]] ], [ 4, [[TRUE_2]] ], [ undef, [[FALSE]] ]
; CHECK-NEXT:    [[R:%.*]] = ashr i32 [[P]], 1
; CHECK-NEXT:    ret i32 [[R]]
;
  br i1 %c.1, label %true.1, label %false

true.1:
  br i1 %c.2, label %true.2, label %exit

true.2:
  br label %exit

false:
  br label %exit

exit:
  %p = phi i32 [ 2, %true.1 ], [ 4, %true.2], [ undef, %false ]
  %r = ashr i32 %p, 1
  ret i32 %r
}
