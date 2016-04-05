; RUN: opt < %s -S -loop-unroll | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define x86_mmx @f() #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %phi = phi i32 [ 1, %entry ], [ %add, %for.body ]
  %add = add i32 %phi, 1
  %cmp = icmp eq i32 %phi, 0
  br i1 %cmp, label %exit, label %for.body

exit:                                             ; preds = %for.body
  %ret = phi x86_mmx [ undef, %for.body ]
  ; CHECK: %[[ret_ph:.*]] = phi x86_mmx [ undef, %entry
  ; CHECK: %[[ret_ph1:.*]]  = phi x86_mmx [ undef,
  ; CHECK: %[[ret:.*]] = phi x86_mmx [ %[[ret_ph]], {{.*}} ], [ %[[ret_ph1]],
  ; CHECK: ret x86_mmx %[[ret]]
  ret x86_mmx %ret
}

attributes #0 = { "target-cpu"="x86-64" }
