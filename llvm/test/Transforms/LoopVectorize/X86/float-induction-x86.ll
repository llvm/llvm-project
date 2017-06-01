; RUN: opt < %s  -O3 -mcpu=core-avx2 -mtriple=x86_64-unknown-linux-gnu -S | FileCheck --check-prefix AUTO_VEC %s

; This test checks auto-vectorization with FP induction variable.
; The FP operation is not "fast" and requires "fast-math" function attribute.

;void fp_iv_loop1(float * __restrict__ A, int N) {
;  float x = 1.0;
;  for (int i=0; i < N; ++i) {
;    A[i] = x;
;    x += 0.5;
;  }
;}


; AUTO_VEC-LABEL: @fp_iv_loop1(
; AUTO_VEC: vector.body
; AUTO_VEC: store <8 x float>

define void @fp_iv_loop1(float* noalias nocapture %A, i32 %N) #0 {
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %x.06 = phi float [ %conv1, %for.body ], [ 1.000000e+00, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %x.06, float* %arrayidx, align 4
  %conv1 = fadd float %x.06, 5.000000e-01
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; The same as the previous, FP operation is not fast, different function attribute
; Vectorization should be rejected.
;void fp_iv_loop2(float * __restrict__ A, int N) {
;  float x = 1.0;
;  for (int i=0; i < N; ++i) {
;    A[i] = x;
;    x += 0.5;
;  }
;}

; AUTO_VEC-LABEL: @fp_iv_loop2(
; AUTO_VEC-NOT: vector.body
; AUTO_VEC-NOT: store <{{.*}} x float>

define void @fp_iv_loop2(float* noalias nocapture %A, i32 %N) #1 {
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %x.06 = phi float [ %conv1, %for.body ], [ 1.000000e+00, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %x.06, float* %arrayidx, align 4
  %conv1 = fadd float %x.06, 5.000000e-01
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

attributes #0 = { "no-nans-fp-math"="true" }
attributes #1 = { "no-nans-fp-math"="false" }
