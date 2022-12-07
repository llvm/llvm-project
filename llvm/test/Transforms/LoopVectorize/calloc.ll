; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK: hexit
;CHECK: zext <4 x i8>
;CHECK: ret

define noalias i8* @hexit(i8* nocapture %bytes, i64 %length) nounwind uwtable ssp {
entry:
  %shl = shl i64 %length, 1
  %add28 = or i64 %shl, 1
  %call = tail call i8* @calloc(i64 1, i64 %add28) nounwind
  %cmp29 = icmp eq i64 %shl, 0
  br i1 %cmp29, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %0 = shl i64 %length, 1
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %i.030 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %shr = lshr i64 %i.030, 1
  %arrayidx = getelementptr inbounds i8, i8* %bytes, i64 %shr
  %1 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %1 to i32
  %and = shl i64 %i.030, 2
  %neg = and i64 %and, 4
  %and3 = xor i64 %neg, 4
  %sh_prom = trunc i64 %and3 to i32
  %shl4 = shl i32 15, %sh_prom
  %and5 = and i32 %conv, %shl4
  %shr11 = lshr i32 %and5, %sh_prom
  %conv13 = and i32 %shr11, 254
  %cmp15 = icmp ugt i32 %conv13, 9
  %cond = select i1 %cmp15, i32 87, i32 48
  %add17 = add nsw i32 %cond, %shr11
  %conv18 = trunc i32 %add17 to i8
  %arrayidx19 = getelementptr inbounds i8, i8* %call, i64 %i.030
  store i8 %conv18, i8* %arrayidx19, align 1
  %inc = add i64 %i.030, 1
  %exitcond = icmp eq i64 %inc, %0
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret i8* %call
}

declare noalias i8* @calloc(i64, i64) nounwind
