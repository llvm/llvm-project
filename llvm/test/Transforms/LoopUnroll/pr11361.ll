; RUN: opt -loop-unroll -disable-output < %s
; PR11361

; This tests for an iterator invalidation issue.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @func_1() nounwind uwtable {
entry:
  br label %for.cond8.preheader

for.cond8.preheader:                              ; preds = %for.inc15, %entry
  %l_1264.04 = phi i32 [ 0, %entry ], [ %add.i, %for.inc15 ]
  %l_1330.0.03 = phi i80 [ undef, %entry ], [ %ins.lcssa, %for.inc15 ]
  br label %for.body9

for.body9:                                        ; preds = %for.body9, %for.cond8.preheader
  %l_1330.0.12 = phi i80 [ %l_1330.0.03, %for.cond8.preheader ], [ %ins, %for.body9 ]
  %storemerge1 = phi i32 [ 7, %for.cond8.preheader ], [ %sub, %for.body9 ]
  %tmp = lshr i80 %l_1330.0.12, 8
  %tmp1 = trunc i80 %tmp to i8
  %inc12 = add i8 %tmp1, 1
  %tmp2 = zext i8 %inc12 to i80
  %tmp3 = shl nuw nsw i80 %tmp2, 8
  %mask = and i80 %l_1330.0.12, -65281
  %ins = or i80 %tmp3, %mask
  %sub = add nsw i32 %storemerge1, -1
  %tobool = icmp eq i32 %sub, 0
  br i1 %tobool, label %for.inc15, label %for.body9

for.inc15:                                        ; preds = %for.body9
  %ins.lcssa = phi i80 [ %ins, %for.body9 ]
  %sext = shl i32 %l_1264.04, 24
  %conv.i = ashr exact i32 %sext, 24
  %add.i = add nsw i32 %conv.i, 1
  %cmp = icmp slt i32 %add.i, 3
  br i1 %cmp, label %for.cond8.preheader, label %for.end16

for.end16:                                        ; preds = %for.inc15
  ret void
}
