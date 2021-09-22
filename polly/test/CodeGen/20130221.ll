; RUN: opt %loadPolly -polly-codegen -S < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @list_sequence(i32* %A) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %i.inc, %for.body ]
  %i.inc = add nsw i32 %i, 1
  %cmp5 = icmp slt i32 %i.inc, 2
  br i1 %cmp5, label %for.body, label %for.next

for.next:
  store i32 %i.inc, i32* %A
  br label %for.end

for.end:
  ret void
}
