; RUN: opt -passes=loop-vectorize -S -mcpu=core-avx2 < %s | FileCheck %s

;float inc = 0.5;
;void foo(ptr A, unsigned N) {
;
;  for (unsigned i=0; i<N; i++){
;    A[i] += inc;
;  }
;}

; CHECK-LABEL: foo
; CHECK: vector.body
; CHECK: load <8 x float>
; CHECK: fadd <8 x float>
; CHECK: store <8 x float>

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@inc = global float 5.000000e-01, align 4

define void @foo(ptr nocapture %A, i32 %N) #0 {
entry:
  %cmp3 = icmp eq i32 %N, 0
  br i1 %cmp3, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %0 = load float, ptr @inc, align 4
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %indvars.iv
  %1 = load float, ptr %arrayidx, align 4
  %add = fadd float %0, %1
  store float %add, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
