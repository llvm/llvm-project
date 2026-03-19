; RUN: opt < %s -passes='loop-vectorize' -enable-epilogue-vectorization -S | FileCheck %s

; TODO: For now test for the `-epilogue-vectorization-minimum-VF` option. In
; the future we need to replace this with a more meaningful test of the
; epilogue vectorization cost-model.
; RUN: opt < %s -passes='loop-vectorize' -enable-epilogue-vectorization -epilogue-vectorization-minimum-VF=4 -S | FileCheck %s --check-prefix=CHECK-MIN-4
; RUN: opt < %s -passes='loop-vectorize' -enable-epilogue-vectorization -force-vector-interleave=1 -S | FileCheck %s --check-prefix=CHECK-MIN-IC-1

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Do not vectorize epilogues for loops with minsize attribute
; CHECK-LABEL: @f1
; CHECK-NOT: vector.main.loop.iter.check
; CHECK-NOT: vec.epilog.iter.check
; CHECK-NOT: vec.epilog.ph
; CHECK-NOT: vec.epilog.vector.body
; CHECK-NOT: vec.epilog.middle.block
; CHECK: ret void

define dso_local void @f1(ptr noalias %aa, ptr noalias %bb, ptr noalias %cc, i32 signext %N) #0 {
entry:
  %cmp1 = icmp sgt i32 %N, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %bb, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %cc, i64 %indvars.iv
  %1 = load float, ptr %arrayidx2, align 4
  %add = fadd fast float %0, %1
  %arrayidx4 = getelementptr inbounds float, ptr %aa, i64 %indvars.iv
  store float %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Do not vectorize epilogues for loops with optsize attribute
; CHECK-LABEL: @f2
; CHECK-NOT: vector.main.loop.iter.check
; CHECK-NOT: vec.epilog.iter.check
; CHECK-NOT: vec.epilog.ph
; CHECK-NOT: vec.epilog.vector.body
; CHECK-NOT: vec.epilog.middle.block
; CHECK: ret void

define dso_local void @f2(ptr noalias %aa, ptr noalias %bb, ptr noalias %cc, i32 signext %N) #1 {
entry:
  %cmp1 = icmp sgt i32 %N, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %bb, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %cc, i64 %indvars.iv
  %1 = load float, ptr %arrayidx2, align 4
  %add = fadd fast float %0, %1
  %arrayidx4 = getelementptr inbounds float, ptr %aa, i64 %indvars.iv
  store float %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Do not vectorize the epilogue for loops with VF*IC less than the default -epilogue-vectorization-minimum-VF of 16.
; CHECK-MIN-IC-1-LABEL: @f3
; CHECK-MIN-IC-1-NOT: vector.main.loop.iter.check
; CHECK-MIN-IC-1-NOT: vec.epilog.iter.check
; CHECK-MIN-IC-1-NOT: vec.epilog.ph
; CHECK-MIN-IC-1-NOT: vec.epilog.vector.body
; CHECK-MIN-IC-1-NOT: vec.epilog.middle.block
; CHECK-MIN-IC-1: ret void

; Specify a smaller minimum VF (via `-epilogue-vectorization-minimum-VF=4`) and
; make sure the epilogue gets vectorized in that case.
; CHECK-MIN-4-LABEL: @f3
; CHECK-MIN-4: vector.main.loop.iter.check
; CHECK-MIN-4: vec.epilog.iter.check
; CHECK-MIN-4: vec.epilog.ph
; CHECK-MIN-4: vec.epilog.vector.body
; CHECK-MIN-4: vec.epilog.middle.block
; CHECK-MIN-4: ret void

; Default behaviour is to vectorize the epilogue for this loop.
; CHECK-LABEL: @f3
; CHECK: vector.main.loop.iter.check
; CHECK: vec.epilog.iter.check
; CHECK: vec.epilog.ph
; CHECK: vec.epilog.vector.body
; CHECK: vec.epilog.middle.block
; CHECK: ret void

define dso_local void @f3(ptr noalias %aa, ptr noalias %bb, ptr noalias %cc, i32 signext %N) {
entry:
  %cmp1 = icmp sgt i32 %N, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %bb, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %cc, i64 %indvars.iv
  %1 = load float, ptr %arrayidx2, align 4
  %add = fadd fast float %0, %1
  %arrayidx4 = getelementptr inbounds float, ptr %aa, i64 %indvars.iv
  store float %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

attributes #0 = { minsize }
attributes #1 = { optsize }
