; RUN: opt %loadPolly -polly-scops -pass-remarks-analysis=.* -disable-output < %s 2>&1 | FileCheck %s

; Make sure we hit the complexity bailout, and don't crash.
; CHECK: Low complexity assumption:       {  : false }

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8--linux-android"

define hidden void @f(i32 %arg1, i32 %arg2, i32 %cond, i32 %tmp, i32 %tmp196) {
entry:
  %div = sdiv i32 %tmp, 8
  %div10 = sdiv i32 %arg1, 8
  %div11 = sdiv i32 %tmp196, 2
  %add = add nsw i32 %div10, %div11
  %sub19 = add nsw i32 %div, -1
  %cmp20 = icmp slt i32 %add, %sub19
  %add.sub19 = select i1 %cmp20, i32 %add, i32 %sub19
  %div469 = sdiv i32 %arg2, 8
  %cmp.i68 = icmp slt i32 %div469, %cond
  %cond.i = select i1 %cmp.i68, i32 %cond, i32 %div469
  %sub.i69 = add i32 0, %div469
  %cmp9.i = icmp sgt i32 %sub.i69, %add.sub19
  %sub15.max_x.i = select i1 %cmp9.i, i32 %add.sub19, i32 %sub.i69
  %sub30.i = sub nsw i32 %sub15.max_x.i, %cond.i
  %add31.i = add nsw i32 %sub30.i, 1
  br label %for.body.us.i

for.body.us.i:
  br label %for.body47.us.i

for.body47.us.i:
  %cmp45.us.i = icmp ult i32 0, %add31.i
  br i1 %cmp45.us.i, label %for.body47.us.i, label %for.cond44.for.cond.cleanup46_crit_edge.us.i

for.cond44.for.cond.cleanup46_crit_edge.us.i:
  br label %for.body.us.i
}
