; RUN: llc < %s
; PR 5300
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

@g_296 = external dso_local global i8, align 1              ; <ptr> [#uses=1]

define noalias ptr @func_31(ptr nocapture %int8p_33, ptr nocapture %p_34, ptr nocapture %p_35) nounwind {
entry:
  %cmp.i = icmp sgt i16 undef, 234                ; <i1> [#uses=1]
  %tmp17 = select i1 %cmp.i, i16 undef, i16 0     ; <i16> [#uses=2]
  %conv8 = trunc i16 %tmp17 to i8                 ; <i8> [#uses=3]
  br i1 undef, label %cond.false.i29, label %land.lhs.true.i

land.lhs.true.i:                                  ; preds = %entry
  %tobool5.i = icmp eq i32 undef, undef           ; <i1> [#uses=1]
  br i1 %tobool5.i, label %cond.false.i29, label %bar.exit

cond.false.i29:                                   ; preds = %land.lhs.true.i, %entry
  %tmp = sub i8 0, %conv8                         ; <i8> [#uses=1]
  %mul.i = and i8 %conv8, %tmp                    ; <i8> [#uses=1]
  br label %bar.exit

bar.exit:                                         ; preds = %cond.false.i29, %land.lhs.true.i
  %call1231 = phi i8 [ %mul.i, %cond.false.i29 ], [ %conv8, %land.lhs.true.i ] ; <i8> [#uses=0]
  %conv21 = trunc i16 %tmp17 to i8                ; <i8> [#uses=1]
  store i8 %conv21, ptr @g_296
  ret ptr undef
}
