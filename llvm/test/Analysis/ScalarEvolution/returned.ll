; RUN: opt < %s -S -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define i8* @foo(i32 %no, i8* nocapture %d) nounwind {
entry:
  %v = call i8* @func1(i8* %d)
  %w = getelementptr i8, i8* %v, i64 5
  ret i8* %w
}

; CHECK-LABEL: Classifying expressions for: @foo
; CHECK: %w = getelementptr i8, i8* %v, i64 5
; CHECK-NEXT: (5 + %d)

declare i8* @func1(i8* returned) nounwind argmemonly

