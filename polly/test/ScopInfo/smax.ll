; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:32-n32-S64"

define void @foo(i32 * noalias %data, i32 * noalias %ptr, i32 %x_pos, i32 %w) {
entry:
  br label %for.body

for.body:
  %x = phi i32 [ 0, %entry ], [ %x.inc, %for.body ]
  %add = add nsw i32 %x, %x_pos
  %cmp1 = icmp sgt i32 %add, %w
  %cond = select i1 %cmp1, i32 %w, i32 %add
  %arrayidx = getelementptr inbounds i32, i32* %ptr, i32 %cond
  store i32 1, i32* %arrayidx
  %x.inc = add nsw i32 %x, 1
  %cmp = icmp slt i32 %x.inc, 2
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; We check that there are only two parameters, but not a third one that
; represents the smax() expression. This test case comes from PR 18155.
; CHECK: [w, x_pos]
