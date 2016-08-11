; RUN: opt < %s -instcombine -S | FileCheck %s

; Induction variable is known to be non-negative
; when its initial value is non-negative and 
; increments by non-negative value
define i32 @test_indvar_nonnegative_add() {
; CHECK-LABEL: @test_indvar_nonnegative_add(
; CHECK: br i1 true, label %for.end, label %for.body
entry:
  br label %for.body

for.body:
  %i = phi i32 [0, %entry], [%inc, %for.body]
  %inc = add nsw i32 %i, 1
  %cmp = icmp sge i32 %i, 0
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %i
}

; Induction variable is known to be non-negative
; when its initial value is non-negative and 
; is multiplied by a non-negative value in each 
; iteration
define i32 @test_indvar_nonnegative_mul() {
; CHECK-LABEL: @test_indvar_nonnegative_mul(
; CHECK: br i1 true, label %for.end, label %for.body
entry:
  br label %for.body

for.body:
  %i = phi i32 [1, %entry], [%inc, %for.body]
  %inc = mul nsw i32 %i, 3
  %cmp = icmp sge i32 %i, 0
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %i
}

; Induction variable is known to be non-negative,
; Similar to add
define i32 @test_indvar_nonnegative_sub(i32 %a) {
; CHECK-LABEL: @test_indvar_nonnegative_sub(
; CHECK: br i1 true, label %for.end, label %for.body
entry:
  br label %for.body

for.body:
  %i = phi i32 [0, %entry], [%inc, %for.body]
  %b = or i32 %a, -2147483648
  %inc = sub nsw i32 %i, %b
  %cmp = icmp sge i32 %i, 0
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %i
}

; Induction variable is known to be negative when 
; its initial value is negative and decrements by
; a non-negative value
define i32 @test_indvar_negative_add() {
; CHECK-LABEL: @test_indvar_negative_add(
; CHECK: br i1 true, label %for.end, label %for.body
entry:
  br label %for.body

for.body:
  %i = phi i32 [-1, %entry], [%inc, %for.body]
  %inc = add nsw i32 %i, -1
  %cmp = icmp slt i32 %i, 0
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %i
}

; Induction variable is known to be negative,
; similar to add
define i32 @test_indvar_negative_sub(i32 %a) {
; CHECK-LABEL: @test_indvar_negative_sub(
; CHECK: br i1 true, label %for.end, label %for.body
entry:
  br label %for.body

for.body:
  %i = phi i32 [-1, %entry], [%inc, %for.body]
  %b = and i32 %a, 2147483647
  %inc = sub nsw i32 %i, %b
  %cmp = icmp slt i32 %i, 0
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %i
}
