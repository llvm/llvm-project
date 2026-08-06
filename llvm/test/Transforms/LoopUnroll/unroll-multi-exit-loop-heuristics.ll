; RUN: opt < %s -passes=loop-unroll -unroll-runtime=true -verify-dom-info -verify-loop-info -unroll-runtime-other-exit-predictable=false -S | FileCheck %s
; RUN: opt < %s -passes=loop-unroll -unroll-runtime=true -verify-dom-info -verify-loop-info -unroll-runtime-multi-exit=false -unroll-runtime-other-exit-predictable=false -S | FileCheck %s -check-prefix=NOUNROLL

; Multi exit loop with predictable exit -- unroll
; Exit before loop body.
define i32 @test1(ptr nocapture %a, i64 %n) {
; CHECK-LABEL: @test1(
; CHECK: epil
;
; NOUNROLL-LABEL: @test1(
; NOUNROLL-NOT: epil
;
entry:
  br label %header

header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %latch ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %latch ], [ 0, %entry ]
  br label %for.exiting_block

for.exiting_block:
  %cmp = icmp eq i64 %n, 42
  br i1 %cmp, label %otherexit, label %latch, !prof !0

latch:
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %latchexit, label %header

latchexit:                                          ; preds = %latch
  %sum.0.lcssa = phi i32 [ %add, %latch ]
  ret i32 %sum.0.lcssa

otherexit:
  %rval = call i32 @foo()
  ret i32 %rval
}

declare i32 @foo()

!0 = !{!"branch_weights", i32 1, i32 100}

; Exit is a deopt call so it should unroll
define i32 @test2(ptr nocapture %a, i64 %n) {
; CHECK-LABEL: @test2(
; CHECK: epil
;
; NOUNROLL-LABEL: @test2(
; NOUNROLL-NOT: epil
;
entry:
  br label %header

header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %latch ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %latch ], [ 0, %entry ]
  br label %for.exiting_block

for.exiting_block:
  %cmp = icmp eq i64 %n, 42
  br i1 %cmp, label %otherexit, label %latch

latch:
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %latchexit, label %header

latchexit:                                          ; preds = %latch
  %sum.0.lcssa = phi i32 [ %add, %latch ]
  ret i32 %sum.0.lcssa

otherexit:
  %rval = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %sum.02) ]
  ret i32 %rval
}

declare i32 @llvm.experimental.deoptimize.i32(...)

; multi exit loop where the exits are not predictable -- no unroll
define i32 @test3(ptr nocapture %a, i64 %n) {
; CHECK-LABEL: @test3(
; CHECK-NOT: epil
;
; NOUNROLL-LABEL: @test3(
; NOUNROLL-NOT: epil
;
entry:
  br label %header

header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %latch ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %latch ], [ 0, %entry ]
  br label %for.exiting_block

for.exiting_block:
  %cmp = icmp eq i64 %n, 42
  br i1 %cmp, label %otherexit, label %latch, !prof !2

latch:
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %latchexit, label %header

latchexit:                                          ; preds = %latch
  %sum.0.lcssa = phi i32 [ %add, %latch ]
  ret i32 %sum.0.lcssa

otherexit:
  %rval = call i32 @foo()
  ret i32 %rval
}

!2 = !{!"branch_weights", i32 1, i32 2}

; Multi exit loop with high predictability of exits -- unroll.
; Exit after loop body.
define i32 @test4(ptr nocapture %a, i64 %n) {
; CHECK-LABEL: @test4(
; CHECK: epil
;
; NOUNROLL-LABEL: @test4(
; NOUNROLL-NOT: epil
entry:
  br label %header

header:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %latch ]
  %sum.02 = phi i32 [ 0, %entry ], [ %add, %latch ]
  br label %otherexitingblock

otherexitingblock:
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %cmp = icmp eq i64 %n, 42
  br i1 %cmp, label %otherexit, label %latch, !prof !3

latch:
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %latchexit, label %header

latchexit:
  %sum.0.lcssa = phi i32 [ %add, %latch ]
  ret i32 %sum.0.lcssa

otherexit:
  %rval = call i32 @foo()
  ret i32 %rval
}

!3 = !{!"branch_weights", i32 1, i32 200}
