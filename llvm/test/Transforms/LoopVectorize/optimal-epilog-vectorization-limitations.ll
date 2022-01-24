; REQUIRES: asserts
; RUN: opt < %s  -passes='loop-vectorize' -force-vector-width=2 -enable-epilogue-vectorization -epilogue-vectorization-force-VF=2 --debug-only=loop-vectorize -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64-v256:256:256-v512:512:512"

; Currently we cannot handle live-out variables that are recurrences.
; CHECK: LV: Checking a loop in "f2"
; CHECK: LEV: Unable to vectorize epilogue because the loop is not a supported candidate.

define signext i32 @f2(i8* noalias %A, i32 signext %n) {
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %A, i64 %indvars.iv
  %0 = load i8, i8* %arrayidx, align 1
  %add = add i8 %0, 1
  %arrayidx3 = getelementptr inbounds i8, i8* %A, i64 %indvars.iv
  store i8 %add, i8* %arrayidx3, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  %inc.lcssa.wide = phi i64 [ %indvars.iv.next, %for.body ]
  %1 = trunc i64 %inc.lcssa.wide to i32
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %i.0.lcssa = phi i32 [ 0, %entry ], [ %1, %for.end.loopexit ]
  ret i32 %i.0.lcssa
}

; Currently we cannot handle widended/truncated inductions.
; CHECK: LV: Checking a loop in "f3"
; CHECK: LEV: Unable to vectorize epilogue because the loop is not a supported candidate.

define void @f3(i8* noalias %A, i32 signext %n) {
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %0 = trunc i64 %indvars.iv to i32
  %conv = trunc i32 %0 to i8
  %arrayidx = getelementptr inbounds i8, i8* %A, i64 %indvars.iv
  store i8 %conv, i8* %arrayidx, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
