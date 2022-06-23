; RUN: llc < %s -verify-machineinstrs -enable-misched -verify-misched
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10"

; This function contains a cmp instruction with two users.
; Hoisting the last use requires trimming the EFLAGS live range to the second.
define void @rdar13353090(ptr %plane, i64 %_x1, i64 %_x2) {
entry:
  %cmp = icmp ult i64 %_x1, %_x2
  %cond = select i1 %cmp, i64 %_x1, i64 %_x2
  %cond10 = select i1 %cmp, i64 %_x2, i64 %_x1
  %0 = load i64, ptr null, align 8
  %cmp16 = icmp ult i64 %cond, %0
  %cmp23 = icmp ugt i64 %cond10, 0
  br i1 %cmp16, label %land.lhs.true21, label %return

land.lhs.true21:                                  ; preds = %entry
  %sub = add i64 %0, -1
  br i1 %cmp23, label %if.then24, label %return

if.then24:                                        ; preds = %land.lhs.true21
  %cmp16.i = icmp ult i64 %cond, %sub
  %cond20.i = select i1 %cmp16.i, i64 %cond, i64 %sub
  %add21.i = add i64 0, %cond20.i
  br label %for.body34.i

for.body34.i:                                     ; preds = %for.inc39.i, %if.then24
  %index.178.i = phi i64 [ %add21.i, %if.then24 ], [ %inc41.i, %for.inc39.i ]
  %arrayidx35.i = getelementptr inbounds i8, ptr %plane, i64 %index.178.i
  %1 = load i8, ptr %arrayidx35.i, align 1
  %tobool36.i = icmp eq i8 %1, 0
  br i1 %tobool36.i, label %for.inc39.i, label %return

for.inc39.i:                                      ; preds = %for.body34.i
  %inc41.i = add i64 %index.178.i, 1
  br i1 undef, label %return, label %for.body34.i

return:                                           ; preds = %for.inc39.i, %for.body34.i, %land.lhs.true21, %entry
  ret void
}
