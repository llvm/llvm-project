; RUN: opt -S -loop-vectorize < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Check that we vectorize only bottom-tested loops.
; This is a reduced testcase from PR21302.
;
; rdar://problem/18886083

%struct.X = type { i32, i16 }
; CHECK-LABEL: @foo(
; CHECK-NOT: vector.body

define void @foo(i32 %n) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %for.body, label %if.end

for.body:
  %iprom = sext i32 %i to i64
  %b = getelementptr inbounds %struct.X* undef, i64 %iprom, i32 1
  store i16 0, i16* %b, align 4
  %inc = add nsw i32 %i, 1
  br label %for.cond

if.end:
  ret void
}
