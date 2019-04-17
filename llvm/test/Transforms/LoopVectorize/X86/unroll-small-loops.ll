; RUN: opt < %s  -loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -force-vector-width=4 -force-vector-interleave=0 -dce -S \
; RUN:   | FileCheck %s --check-prefix=CHECK-VECTOR
; RUN: opt < %s  -loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -force-vector-width=1 -force-vector-interleave=0 -dce -S \
; RUN:   | FileCheck %s --check-prefix=CHECK-SCALAR

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; We don't unroll this loop because it has a small constant trip count.
;
; CHECK-VECTOR-LABEL: @foo(
; CHECK-VECTOR: load <4 x i32>
; CHECK-VECTOR-NOT: load <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR-NOT: store <4 x i32>
; CHECK-VECTOR: ret
;
; CHECK-SCALAR-LABEL: @foo(
; CHECK-SCALAR: load i32, i32*
; CHECK-SCALAR-NOT: load i32, i32*
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define i32 @foo(i32* nocapture %A) nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %3 = load i32, i32* %2, align 4
  %4 = add nsw i32 %3, 6
  store i32 %4, i32* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 100
  br i1 %exitcond, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 undef
}

; But this is a good small loop to unroll as we don't know of a bound on its
; trip count.
;
; CHECK-VECTOR-LABEL: @bar(
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: store <4 x i32>
; CHECK-VECTOR: ret
;
; For x86, loop unroll in loop vectorizer is disabled when VF==1.
;
; CHECK-SCALAR-LABEL: @bar(
; CHECK-SCALAR: store i32
; CHECK-SCALAR-NOT: store i32
; CHECK-SCALAR: ret
define i32 @bar(i32* nocapture %A, i32 %n) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %3 = load i32, i32* %2, align 4
  %4 = add nsw i32 %3, 6
  store i32 %4, i32* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}

; Also unroll if we need a runtime check but it was going to be added for
; vectorization anyways.
; CHECK-VECTOR-LABEL: @runtime_chk(
; CHECK-VECTOR: store <4 x float>
; CHECK-VECTOR: store <4 x float>
;
; But not if the unrolling would introduce the runtime check.
; CHECK-SCALAR-LABEL: @runtime_chk(
; CHECK-SCALAR: store float
; CHECK-SCALAR-NOT: store float
define void @runtime_chk(float* %A, float* %B, float %N) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %B, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %mul = fmul float %0, %N
  %arrayidx2 = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %mul, float* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
