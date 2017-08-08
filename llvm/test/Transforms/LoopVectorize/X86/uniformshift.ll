; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -loop-vectorize -debug-only=loop-vectorize -S < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: "foo"
; CHECK: LV: Found an estimated cost of 1 for VF 4 For instruction:   %shift = ashr i32 %val, %k
define void @foo(i32* nocapture %p, i32 %k) local_unnamed_addr #0 {
entry:  
  br label %body

body:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %ptr = getelementptr inbounds i32, i32* %p, i64 %i
  %val = load i32, i32* %ptr, align 4
  %shift = ashr i32 %val, %k
  store i32 %shift, i32* %ptr, align 4
  %next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %next, 16
  br i1 %cmp, label %exit, label %body

exit:
  ret void

}
