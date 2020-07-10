; RUN: opt < %s -S -loop-unroll -unroll-max-iteration-count-to-analyze=100 -unroll-threshold=10 -unroll-max-percent-threshold-boost=200 | FileCheck %s
; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop(loop-unroll-full)' -unroll-max-iteration-count-to-analyze=100 -unroll-threshold=10 -unroll-max-percent-threshold-boost=200 | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@known_constant = internal unnamed_addr constant [10 x i32] [i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1], align 16

; If we can figure out result of comparison on each iteration, we can resolve
; the depending branch. That means, that the unrolled version of the loop would
; have less code, because we don't need not-taken basic blocks there.
; This test checks that this is taken into consideration.
; We expect this loop to be unrolled, because the most complicated part of its
; body (if.then block) is never actually executed.
; CHECK-LABEL: @branch_folded
; CHECK-NOT: br i1 %
; CHECK: ret i32
define i32 @branch_folded(i32* noalias nocapture readonly %b) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %iv.0 = phi i64 [ 0, %entry ], [ %iv.1, %for.inc ]
  %r.0 = phi i32 [ 0, %entry ], [ %r.1, %for.inc ]
  %arrayidx1 = getelementptr inbounds [10 x i32], [10 x i32]* @known_constant, i64 0, i64 %iv.0
  %x1 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp eq i32 %x1, 0
  %iv.1 = add nuw nsw i64 %iv.0, 1
  br i1 %cmp, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv.0
  %x2 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %x2, %r.0
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %r.1 = phi i32 [ %add, %if.then ], [ %x1, %for.body ]
  %exitcond = icmp eq i64 %iv.1, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  ret i32 %r.1
}

; Check that we don't crash when we analyze icmp with pointer-typed IV and a
; pointer.
; CHECK-LABEL: @ptr_cmp_crash
; CHECK:   ret void
define void @ptr_cmp_crash() {
entry:
  br label %while.body

while.body:
  %iv.0 = phi i32* [ getelementptr inbounds ([10 x i32], [10 x i32]* @known_constant, i64 0, i64 0), %entry ], [ %iv.1, %while.body ]
  %iv.1 = getelementptr inbounds i32, i32* %iv.0, i64 1
  %exitcond = icmp eq i32* %iv.1, getelementptr inbounds ([10 x i32], [10 x i32]* @known_constant, i64 0, i64 9)
  br i1 %exitcond, label %loop.exit, label %while.body

loop.exit:
  ret void
}

; Check that we don't crash when we analyze ptrtoint cast.
; CHECK-LABEL: @ptrtoint_cast_crash
; CHECK:   ret void
define void @ptrtoint_cast_crash(i8 * %a) {
entry:
  %limit = getelementptr i8, i8* %a, i64 512
  br label %loop.body

loop.body:
  %iv.0 = phi i8* [ %a, %entry ], [ %iv.1, %loop.body ]
  %cast = ptrtoint i8* %iv.0 to i64
  %iv.1 = getelementptr inbounds i8, i8* %iv.0, i64 1
  %exitcond = icmp ne i8* %iv.1, %limit
  br i1 %exitcond, label %loop.body, label %loop.exit

loop.exit:
  ret void
}
