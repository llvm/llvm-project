; RUN: llc < %s -mtriple=x86_64-- >/dev/null

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @test0() nounwind {
if.end90.i.i:
  br label %while.body.i.i221.i

while.body.i.i221.i:                              ; preds = %while.cond.backedge.i.i.i, %if.end90.i.i
  br i1 undef, label %if.then.i.i224.i, label %while.cond.backedge.i.i.i

while.cond.backedge.i.i.i:                        ; preds = %for.end.i.i.i, %while.body.i.i221.i
  br label %while.body.i.i221.i

if.then.i.i224.i:                                 ; preds = %while.body.i.i221.i
  switch i32 undef, label %for.cond.i.i226.i [
    i32 92, label %sw.bb.i.i225.i
    i32 34, label %sw.bb.i.i225.i
    i32 110, label %sw.bb21.i.i.i
  ]

sw.bb.i.i225.i:                                   ; preds = %if.then.i.i224.i, %if.then.i.i224.i
  unreachable

sw.bb21.i.i.i:                                    ; preds = %if.then.i.i224.i
  unreachable

for.cond.i.i226.i:                                ; preds = %for.body.i.i.i, %if.then.i.i224.i
  %0 = phi i64 [ %tmp154.i.i.i, %for.body.i.i.i ], [ 0, %if.then.i.i224.i ] ; <i64> [#uses=2]
  %tmp154.i.i.i = add i64 %0, 1                   ; <i64> [#uses=2]
  %i.0.i.i.i = trunc i64 %0 to i32                ; <i32> [#uses=1]
  br i1 undef, label %land.rhs.i.i.i, label %for.end.i.i.i

land.rhs.i.i.i:                                   ; preds = %for.cond.i.i226.i
  br i1 undef, label %for.body.i.i.i, label %for.end.i.i.i

for.body.i.i.i:                                   ; preds = %land.rhs.i.i.i
  br label %for.cond.i.i226.i

for.end.i.i.i:                                    ; preds = %land.rhs.i.i.i, %for.cond.i.i226.i
  %idx.ext.i.i.i = sext i32 %i.0.i.i.i to i64     ; <i64> [#uses=1]
  %sub.ptr72.sum.i.i.i = xor i64 %idx.ext.i.i.i, -1 ; <i64> [#uses=1]
  %pos.addr.1.sum155.i.i.i = add i64 %tmp154.i.i.i, %sub.ptr72.sum.i.i.i ; <i64> [#uses=1]
  %arrayidx76.i.i.i = getelementptr inbounds i8, ptr undef, i64 %pos.addr.1.sum155.i.i.i ; <ptr> [#uses=0]
  br label %while.cond.backedge.i.i.i
}

define void @test1() nounwind {
entry:
  %t = shl i32 undef, undef                     ; <i32> [#uses=1]
  %t9 = sub nsw i32 0, %t                     ; <i32> [#uses=1]
  br label %outer

outer:                                             ; preds = %bb18, %bb
  %i12 = phi i32 [ %t21, %bb18 ], [ 0, %entry ]  ; <i32> [#uses=2]
  %i13 = phi i32 [ %t20, %bb18 ], [ 0, %entry ]  ; <i32> [#uses=2]
  br label %inner

inner:                                             ; preds = %bb16, %bb11
  %t17 = phi i32 [ %i13, %outer ], [ undef, %inner ] ; <i32> [#uses=1]
  store i32 %t17, ptr undef
  br i1 undef, label %bb18, label %inner

bb18:                                             ; preds = %bb16
  %t19 = add i32 %i13, %t9                 ; <i32> [#uses=1]
  %t20 = add i32 %t19, %i12                 ; <i32> [#uses=1]
  %t21 = add i32 %i12, 1                      ; <i32> [#uses=1]
  br label %outer
}
