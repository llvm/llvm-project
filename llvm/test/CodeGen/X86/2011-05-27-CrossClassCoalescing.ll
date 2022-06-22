; RUN: llc < %s -verify-coalescing
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.0"

@bit_count = external constant [256 x i32], align 16

define fastcc void @unate_intersect() nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc.i
  br label %do.body.i

do.body.i:                                        ; preds = %do.body.i, %for.body
  %exitcond149 = icmp eq i64 undef, undef
  br i1 %exitcond149, label %land.lhs.true, label %do.body.i

land.lhs.true:                                    ; preds = %do.body.i
  br label %for.body.i

for.body.i:                                       ; preds = %for.inc.i, %if.then
  %tmp3524.i = phi i32 [ 0, %land.lhs.true ], [ %tmp351.i, %for.inc.i ]
  %tmp6.i12 = load i32, ptr undef, align 4
  br i1 undef, label %for.inc.i, label %if.then.i17

if.then.i17:                                      ; preds = %for.body.i
  %shr.i14 = lshr i32 %tmp6.i12, 8
  %and14.i = and i32 %shr.i14, 255
  %idxprom15.i = zext i32 %and14.i to i64
  %arrayidx16.i = getelementptr inbounds [256 x i32], ptr @bit_count, i64 0, i64 %idxprom15.i
  %tmp17.i15 = load i32, ptr %arrayidx16.i, align 4
  %add.i = add i32 0, %tmp3524.i
  %add24.i = add i32 %add.i, %tmp17.i15
  %add31.i = add i32 %add24.i, 0
  %add33.i = add i32 %add31.i, 0
  br label %for.inc.i

for.inc.i:                                        ; preds = %if.then.i17, %for.body.i
  %tmp351.i = phi i32 [ %add33.i, %if.then.i17 ], [ %tmp3524.i, %for.body.i ]
  br label %for.body.i
}
