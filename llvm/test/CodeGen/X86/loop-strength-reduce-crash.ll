; RUN: opt -loop-reduce < %s -o /dev/null

; LSR doesn't actually do anything on this input; just check that it doesn't
; crash while building the compatible type for the IV (by virtue of using
; INT64_MAX as a constant in the loop).

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define void @foo() {
entry:
  br label %for

for: 
  %0 = phi i64 [ %add, %for ], [ undef, %entry ]
  %next = phi i32 [ %inc, %for ], [ undef, %entry ]
  store i32 %next, ptr undef, align 4
  %add = add i64 %0, 9223372036854775807
  %inc = add nsw i32 %next, 1
  br i1 undef, label %exit, label %for

exit:
  store i64 %add, ptr undef
  ret void
}
