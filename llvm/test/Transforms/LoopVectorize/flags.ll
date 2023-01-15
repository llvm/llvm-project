; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK-LABEL: @flags1(
;CHECK: load <4 x i32>
;CHECK: mul nsw <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret i32
define i32 @flags1(i32 %n, ptr nocapture %A) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 9
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 9, %0 ]
  %2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %3 = load i32, ptr %2, align 4
  %4 = mul nsw i32 %3, 3
  store i32 %4, ptr %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}


;CHECK-LABEL: @flags2(
;CHECK: load <4 x i32>
;CHECK: mul <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret i32
define i32 @flags2(i32 %n, ptr nocapture %A) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 9
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 9, %0 ]
  %2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %3 = load i32, ptr %2, align 4
  %4 = mul i32 %3, 3
  store i32 %4, ptr %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}

; Make sure we copy fast math flags and use them for the final reduction.
; CHECK-LABEL: fast_math
; CHECK: load <4 x float>
; CHECK: fadd fast <4 x float>
; CHECK: br
; CHECK: call fast float @llvm.vector.reduce.fadd.v4f32
define float @fast_math(ptr noalias %s) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %q.04 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %s, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd fast float %q.04, %0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  %add.lcssa = phi float [ %add, %for.body ]
  ret float %add.lcssa
}
