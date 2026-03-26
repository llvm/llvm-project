; RUN: opt < %s -passes=loop-vectorize -mattr=avx,+slow-unaligned-mem-32 -S | FileCheck %s --check-prefix=SLOWMEM32 --check-prefix=CHECK
; RUN: opt < %s -passes=loop-vectorize -mattr=avx,-slow-unaligned-mem-32 -S | FileCheck %s --check-prefix=FASTMEM32 --check-prefix=CHECK

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK-LABEL: @read_mod_write_single_ptr(
; CHECK: load <8 x float>
; CHECK: ret void
define void @read_mod_write_single_ptr(ptr nocapture %a, i32 %n) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %3 = load float, ptr %2, align 4
  %4 = fmul float %3, 3.000000e+00
  store float %4, ptr %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}


; CHECK-LABEL: @read_mod_i64(
; SLOWMEM32: load <2 x i64>
; FASTMEM32: load <4 x i64>
; CHECK: ret void
define void @read_mod_i64(ptr nocapture %a, i32 %n) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %3 = load i64, ptr %2, align 4
  %4 = add i64 %3, 3
  store i64 %4, ptr %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}

