; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S -inline-threshold=50 | FileCheck %s

; This tests that the function count of a function gets properly scaled after 
; inlining a call chain leading to the function.
; Function a calls c with count 200 (C1)
; Function c calls e with count 250 (C2)
; Entry count of e is 500 (C3)
; Entry count of c is 500 (C4)
; Function b calls c with count 300 (C5)
; c->e inlining does not happen since the cost exceeds threshold.
; c then inlined into a.
; e now gets inlined into a (through c) since the branch condition in e is now
; known and hence the cost gets reduced.
; Estimated count of a->e callsite = C2 * (C1 / C4)
; Estimated count of a->e callsite = 250 * (200 / 500) = 100
; Remaining count of e = C3 - 100 = 500 - 100 = 400
; Remaining count of c = C4 - C1 - C5 = 500 - 200 - 300 = 0

@data = external global i32

define i32 @a(i32 %a1) !prof !1 {
  %a2 = call i32 @c(i32 %a1, i32 1)
  ret i32 %a2
}

define i32 @b(i32 %b1) !prof !2 {
  %b2 = call i32 @c(i32 %b1, i32 %b1)
  ret i32 %b2
}

declare void @ext();

; CHECK: @c(i32 %c1, i32 %c100) !prof [[COUNT1:![0-9]+]]
define i32 @c(i32 %c1, i32 %c100) !prof !3 {
  call void @ext()
  %cond = icmp sle i32 %c1, 1
  br i1 %cond, label %cond_true, label %cond_false

cond_false:
  ret i32 0

cond_true:
  %c11 = call i32 @e(i32 %c100)
  ret i32 %c11
}


; CHECK: @e(i32 %c1) !prof [[COUNT2:![0-9]+]]
define i32 @e(i32 %c1) !prof !4 {
  %cond = icmp sle i32 %c1, 1
  br i1 %cond, label %cond_true, label %cond_false

cond_false:
  call void @ext()
  %c2 = load i32, i32* @data, align 4
  %c3 = add i32 %c1, %c2
  %c4 = mul i32 %c3, %c2
  %c5 = add i32 %c4, %c2
  %c6 = mul i32 %c5, %c2
  %c7 = add i32 %c6, %c2
  %c8 = mul i32 %c7, %c2
  %c9 = add i32 %c8, %c2
  %c10 = mul i32 %c9, %c2
  ret i32 %c10

cond_true:
  ret i32 0
}

!llvm.module.flags = !{!0}
; CHECK: [[COUNT1]] = !{!"function_entry_count", i64 0}
; CHECK: [[COUNT2]] = !{!"function_entry_count", i64 400}
!0 = !{i32 1, !"MaxFunctionCount", i32 5000}
!1 = !{!"function_entry_count", i64 200}
!2 = !{!"function_entry_count", i64 300}
!3 = !{!"function_entry_count", i64 500}
!4 = !{!"function_entry_count", i64 500}

