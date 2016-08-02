; RUN: opt < %s -S -loop-unroll -unroll-max-iteration-count-to-analyze=100 -unroll-dynamic-cost-savings-discount=1000 -unroll-threshold=10 -unroll-percent-dynamic-cost-saved-threshold=50 | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@known_constant = internal unnamed_addr constant [10 x i32] [i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1], align 16

; We should be able to propagate constant data through comparisons.
; For example, in this test we have a load, which becomes constant after
; unrolling, making comparison with 0 also known to be 0 (false) - and that
; will trigger further simplifications.
;
; We expect this loop to be unrolled, because in this case load would become
; constant, which is always 1, and which, in its turn, helps to simplify
; following comparison, zero-extension, and addition. In total, unrolling should help to
; optimize more than 50% of all instructions in this case.
;
; CHECK-LABEL: @const_compare
; CHECK-NOT: br i1 %
; CHECK: ret i32
define i32 @const_compare(i32* noalias nocapture readonly %b) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %iv.0 = phi i64 [ 0, %entry ], [ %iv.1, %for.body ]
  %r.0 = phi i32 [ 0, %entry ], [ %r.1, %for.body ]
  %arrayidx1 = getelementptr inbounds [10 x i32], [10 x i32]* @known_constant, i64 0, i64 %iv.0
  %x1 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp eq i32 %x1, 0
  %cast = zext i1 %cmp to i32
  %iv.1 = add nuw nsw i64 %iv.0, 1
  %r.1 = add i32 %r.0, %cast
  %exitcond = icmp eq i64 %iv.1, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  ret i32 %r.1
}

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

; This test is similar to the previous one, but in this we use IV in comparison
; (not a loaded value as we did there).
; CHECK-LABEL: @branch_iv
; CHECK-NOT: br i1 %
; CHECK: ret i64
define i64 @branch_iv(i64* noalias nocapture readonly %b) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %tmp3, %for.inc ]
  %r.030 = phi i64 [ 0, %entry ], [ %r.1, %for.inc ]
  %cmp3 = icmp eq i64 %indvars.iv, 5
  %tmp3 = add nuw nsw i64 %indvars.iv, 1
  br i1 %cmp3, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i64, i64* %b, i64 %tmp3
  %tmp1 = load i64, i64* %arrayidx2, align 4
  %add = add nsw i64 %tmp1, %r.030
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %r.1 = phi i64 [ %add, %if.then ], [ %r.030, %for.body ]
  %exitcond = icmp eq i64 %tmp3, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  ret i64 %r.1
}

; Induction variables are often casted to another type, and that shouldn't
; prevent us from folding branches. Tthis test specifically checks if we can
; handle this. Other than thatm it's similar to the previous test.
; CHECK-LABEL: @branch_iv_trunc
; CHECK-NOT:   br i1 %
; CHECK:   ret i32
define i32 @branch_iv_trunc(i32* noalias nocapture readonly %b) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %tmp3, %for.inc ]
  %r.030 = phi i32 [ 0, %entry ], [ %r.1, %for.inc ]
  %tmp2 = trunc i64 %indvars.iv to i32
  %cmp3 = icmp eq i32 %tmp2, 5
  %tmp3 = add nuw nsw i64 %indvars.iv, 1
  br i1 %cmp3, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %tmp3
  %tmp1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %tmp1, %r.030
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %r.1 = phi i32 [ %add, %if.then ], [ %r.030, %for.body ]
  %exitcond = icmp eq i64 %tmp3, 10
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

; Loop unroller should be able to predict that a comparison would become
; constant if the operands are pointers with the same base and constant
; offsets.
; We expect this loop to be unrolled, since most of its instructions would
; become constant after it.
; CHECK-LABEL: @ptr_cmp
; CHECK-NOT:   br i1 %
; CHECK:   ret i64
define i64 @ptr_cmp(i8 * %a) {
entry:
  %limit = getelementptr i8, i8* %a, i64 40
  %start.iv2 = getelementptr i8, i8* %a, i64 7
  br label %loop.body

loop.body:
  %iv.0 = phi i8* [ %a, %entry ], [ %iv.1, %loop.body ]
  %iv2.0 = phi i8* [ %start.iv2, %entry ], [ %iv2.1, %loop.body ]
  %r.0 = phi i64 [ 0, %entry ], [ %r.1, %loop.body ]
  %cast = ptrtoint i8* %iv.0 to i64
  %cmp = icmp eq i8* %iv2.0, %iv.0
  %sub = sext i1 %cmp to i64
  %mul = mul i64 %sub, %cast
  %r.1 = add i64 %r.0, %mul
  %iv.1 = getelementptr inbounds i8, i8* %iv.0, i64 1
  %iv2.1 = getelementptr inbounds i8, i8* %iv2.0, i64 1
  %exitcond = icmp ne i8* %iv.1, %limit
  br i1 %exitcond, label %loop.body, label %loop.exit

loop.exit:
  ret i64 %r.1
}
