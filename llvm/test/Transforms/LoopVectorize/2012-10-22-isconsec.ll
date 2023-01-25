; RUN: opt < %s -passes=loop-vectorize,dce -force-vector-interleave=1 -force-vector-width=4

; Check that we don't crash.

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

module asm "\09.ident\09\22GCC: (GNU) 4.6.3 LLVM: 3.2svn\22"

@b = common global [32000 x float] zeroinitializer, align 16

define i32 @set1ds(i32 %_n, ptr nocapture %arr, float %value, i32 %stride) nounwind uwtable {
entry:
  %0 = icmp sgt i32 %_n, 0
  br i1 %0, label %"3.lr.ph", label %"5"

"3.lr.ph":                                        ; preds = %entry
  %1 = sext i32 %stride to i64
  br label %"3"

"3":                                              ; preds = %"3.lr.ph", %"3"
  %indvars.iv = phi i64 [ 0, %"3.lr.ph" ], [ %indvars.iv.next, %"3" ]
  %2 = shl nsw i64 %indvars.iv, 2
  %3 = getelementptr inbounds i8, ptr %arr, i64 %2
  store float %value, ptr %3, align 4
  %indvars.iv.next = add i64 %indvars.iv, %1
  %4 = trunc i64 %indvars.iv.next to i32
  %5 = icmp slt i32 %4, %_n
  br i1 %5, label %"3", label %"5"

"5":                                              ; preds = %"3", %entry
  ret i32 0
}

define i32 @init(ptr nocapture %name) unnamed_addr nounwind uwtable {
entry:
  br label %"3"

"3":                                              ; preds = %"3", %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %"3" ]
  %0 = shl nsw i64 %indvars.iv, 2
  %1 = getelementptr inbounds i8, ptr getelementptr inbounds ([32000 x float], ptr @b, i64 0, i64 16000), i64 %0
  store float -1.000000e+00, ptr %1, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 16000
  br i1 %exitcond, label %"5", label %"3"

"5":                                              ; preds = %"3"
  ret i32 0
}
