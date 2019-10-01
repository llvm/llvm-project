; RUN: llc -O1 < %s
; PR12138
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.7.0"

%struct.S0 = type { i8, i32 }

@d = external global [2 x [2 x %struct.S0]], align 4
@c = external global i32, align 4
@e = external global i32, align 4
@b = external global i32, align 4
@a = external global i32, align 4

define void @fn2() nounwind optsize ssp {
entry:
  store i64 0, i64* bitcast ([2 x [2 x %struct.S0]]* @d to i64*), align 4
  %0 = load i32, i32* @c, align 4
  %tobool2 = icmp eq i32 %0, 0
  %1 = load i32, i32* @a, align 4
  %tobool4 = icmp eq i32 %1, 0
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %f.1.0 = phi i32 [ undef, %entry ], [ %sub, %if.end ]
  %g.0 = phi i64 [ 0, %entry ], [ %ins, %if.end ]
  %tobool = icmp eq i32 %f.1.0, 0
  br i1 %tobool, label %for.end, label %for.body

for.body:                                         ; preds = %for.cond
  %2 = lshr i64 %g.0, 32
  %conv = trunc i64 %2 to i16
  br i1 %tobool2, label %lor.rhs, label %lor.end

lor.rhs:                                          ; preds = %for.body
  store i32 1, i32* @e, align 4
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %for.body
  %xor.i = xor i16 %conv, 1
  %p1.lobit.i8 = lshr i64 %g.0, 47
  %p1.lobit.i8.tr = trunc i64 %p1.lobit.i8 to i16
  %p1.lobit.i = and i16 %p1.lobit.i8.tr, 1
  %and.i = and i16 %p1.lobit.i, %xor.i
  %3 = xor i16 %and.i, 1
  %sub.conv.i = sub i16 %conv, %3
  %conv3 = sext i16 %sub.conv.i to i32
  store i32 %conv3, i32* @b, align 4
  br i1 %tobool4, label %if.end, label %for.end

if.end:                                           ; preds = %lor.end
  %mask = and i64 %g.0, -256
  %ins = or i64 %mask, 1
  %sub = add nsw i32 %f.1.0, -1
  br label %for.cond

for.end:                                          ; preds = %lor.end, %for.cond
  ret void
}
