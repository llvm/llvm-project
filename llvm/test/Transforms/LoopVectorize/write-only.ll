; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK-LABEL: @read_mod_write_single_ptr(
;CHECK: load <4 x float>
;CHECK: ret i32
define i32 @read_mod_write_single_ptr(float* nocapture %a, i32 %n) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %3 = load float, float* %2, align 4
  %4 = fmul float %3, 3.000000e+00
  store float %4, float* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}

; Ensure that volatile stores are not vectorized.
; CHECK-LABEL: @read_mod_write_single_ptr_volatile_store(
; CHECK-NOT: store <4 x float>
; CHECK: ret i32
define i32 @read_mod_write_single_ptr_volatile_store(float* nocapture %a, i32 %n) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %3 = load float, float* %2, align 4
  %4 = fmul float %3, 3.000000e+00
  store volatile float %4, float* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}
