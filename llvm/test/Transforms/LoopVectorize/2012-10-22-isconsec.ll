; RUN: opt < %s  -loop-vectorize -dce -force-vector-width=4 

; Check that we don't crash.

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

module asm "\09.ident\09\22GCC: (GNU) 4.6.3 LLVM: 3.2svn\22"

@b = common global [32000 x float] zeroinitializer, align 16

define i32 @set1ds(i32 %_n, float* nocapture %arr, float %value, i32 %stride) nounwind uwtable {
entry:
  %0 = icmp sgt i32 %_n, 0
  br i1 %0, label %"3.lr.ph", label %"5"

"3.lr.ph":                                        ; preds = %entry
  %1 = bitcast float* %arr to i8*
  %2 = sext i32 %stride to i64
  br label %"3"

"3":                                              ; preds = %"3.lr.ph", %"3"
  %indvars.iv = phi i64 [ 0, %"3.lr.ph" ], [ %indvars.iv.next, %"3" ]
  %3 = shl nsw i64 %indvars.iv, 2
  %4 = getelementptr inbounds i8* %1, i64 %3
  %5 = bitcast i8* %4 to float*
  store float %value, float* %5, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, %2
  %6 = trunc i64 %indvars.iv.next to i32
  %7 = icmp slt i32 %6, %_n
  br i1 %7, label %"3", label %"5"

"5":                                              ; preds = %"3", %entry
  ret i32 0
}

define i32 @init(i8* nocapture %name) unnamed_addr nounwind uwtable {
entry:
  br label %"3"

"3":                                              ; preds = %"3", %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %"3" ]
  %0 = shl nsw i64 %indvars.iv, 2
  %1 = getelementptr inbounds i8* bitcast (float* getelementptr inbounds ([32000 x float]* @b, i64 0, i64 16000) to i8*), i64 %0
  %2 = bitcast i8* %1 to float*
  store float -1.000000e+00, float* %2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 16000
  br i1 %exitcond, label %"5", label %"3"

"5":                                              ; preds = %"3"
  ret i32 0
}

!0 = metadata !{metadata !"alias set 7: float", metadata !1}
!1 = metadata !{metadata !1}
