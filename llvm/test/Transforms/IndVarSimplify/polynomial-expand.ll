; RUN: opt < %s -indvars -disable-output
; PR5073

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @ctpmv_(float* noalias nocapture %tmp4, i32 %tmp21) nounwind {
bb20:                                             ; preds = %bb19
  br label %bb24

bb24:                                             ; preds = %bb40, %bb23
  %tmp25 = phi i32 [ %tmp43, %bb40 ], [ %tmp21, %bb20 ] ; <i32> [#uses=4]
  %tmp26 = phi i32 [ %tmp41, %bb40 ], [ undef, %bb20 ] ; <i32> [#uses=2]
  %tmp27 = add nsw i32 %tmp26, -1                 ; <i32> [#uses=1]
  %tmp28 = add nsw i32 %tmp25, -1                 ; <i32> [#uses=2]
  %tmp29 = icmp sgt i32 %tmp28, 0                 ; <i1> [#uses=1]
  br i1 %tmp29, label %bb30, label %bb40

bb30:                                             ; preds = %bb30, %bb24
  %tmp31 = phi i32 [ %tmp39, %bb30 ], [ %tmp28, %bb24 ] ; <i32> [#uses=2]
  %tmp32 = phi i32 [ %tmp37, %bb30 ], [ %tmp27, %bb24 ] ; <i32> [#uses=2]
  %tmp33 = sext i32 %tmp32 to i64                 ; <i64> [#uses=1]
  %tmp35 = getelementptr float* %tmp4, i64 %tmp33 ; <%0*> [#uses=1]
  %tmp36 = load float* %tmp35, align 4               ; <%0> [#uses=0]
  %tmp37 = add nsw i32 %tmp32, -1                 ; <i32> [#uses=1]
  %tmp39 = add nsw i32 %tmp31, -1                 ; <i32> [#uses=1]
  %tmp38 = icmp eq i32 %tmp31, 1                  ; <i1> [#uses=1]
  br i1 %tmp38, label %bb40, label %bb30

bb40:                                             ; preds = %bb30, %bb24
  %tmp41 = sub i32 %tmp26, %tmp25                 ; <i32> [#uses=1]
  %tmp43 = add nsw i32 %tmp25, -1                 ; <i32> [#uses=1]
  %tmp42 = icmp eq i32 %tmp25, 1                  ; <i1> [#uses=1]
  br i1 %tmp42, label %bb46, label %bb24

bb46:                                             ; preds = %bb40, %bb23, %bb19
  ret void
}
