; RUN: opt -mtriple=amdgcn-- -O1 -S -inline-threshold=1 %s | FileCheck %s

; CHECK: @add1alias = alias i32 (i32), ptr @add1
; CHECK: @add1alias2 = alias i32 (i32), ptr @add1

@add1alias = alias i32 (i32), ptr @add1
@add1alias2 = alias i32 (i32), ptr @add1

define i32 @add1(i32) {
  %2 = add nsw i32 %0, 1
  ret i32 %2
}
