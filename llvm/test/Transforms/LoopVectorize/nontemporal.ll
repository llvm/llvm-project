; RUN: opt < %s -passes=loop-vectorize,instcombine -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: @foo(
define void @foo(float* noalias %a, float* noalias %b, float* noalias %c, i32 %N) {
entry:
  %cmp.4 = icmp sgt i32 %N, 0
  br i1 %cmp.4, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]

; Check that we don't lose !nontemporal hint when attempting vectorizing of loads.
; CHECK: load {{.*}} align 4, !nontemporal !0
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4, !nontemporal !0

; Check that we don't introduce !nontemporal hint when the original scalar loads didn't have it.
; CHECK: load {{.*}} align 4{{$}}
  %arrayidx2 = getelementptr inbounds float, float* %c, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %add = fadd float %0, %1

; Check that we don't lose !nontemporal hint when attempting vectorizing of stores.
; CHECK: store {{.*}} align 4, !nontemporal !0
  %arrayidx4 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  store float %add, float* %arrayidx4, align 4, !nontemporal !0

  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
; CHECK: ret void
  ret void
}

!0 = !{i32 1}
