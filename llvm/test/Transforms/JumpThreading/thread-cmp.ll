; RUN: opt -S -passes=jump-threading %s | FileCheck %s
; When simplify a branch based on LVI predicates, we should replace the
; comparison itself with a constant (when possible) in case it's otherwise used.

define i32 @test(ptr %p) {
; CHECK-LABEL: @test
; CHECK: icmp eq
; CHECK-NEXT: br i1 %cmp, label %exit2, label %exit1
; CHECK-NOT: icmp ne
entry:
  %cmp = icmp eq ptr %p, null
  br i1 %cmp, label %is_null, label %not_null
is_null:
  %cmp2 = icmp ne ptr %p, null
  br i1 %cmp2, label %exit1, label %exit2
not_null:
  %cmp3 = icmp ne ptr %p, null
  br i1 %cmp3, label %exit1, label %exit2
exit1:
  ret i32 0
exit2:
  ret i32 1
}

declare void @use(i1)

; It would not be legal to replace %cmp2 (well, in this case it actually is,
; but that's a CSE problem, not a LVI/jump threading problem)
define i32 @test_negative(ptr %p) {
; CHECK-LABEL: @test
; CHECK: icmp ne
; CHECK: icmp eq
; CHECK-NEXT: br i1 %cmp, label %exit2, label %exit1
; CHECK-NOT: icmp ne
entry:
  %cmp2 = icmp ne ptr %p, null
  call void @use(i1 %cmp2)
  %cmp = icmp eq ptr %p, null
  br i1 %cmp, label %is_null, label %not_null
is_null:
  br i1 %cmp2, label %exit1, label %exit2
not_null:
  br i1 %cmp2, label %exit1, label %exit2
exit1:
  ret i32 0
exit2:
  ret i32 1
}

; In this case, we can remove cmp2 because it's otherwise unused
define i32 @test2(ptr %p) {
; CHECK-LABEL: @test
; CHECK-LABEL: entry:
; CHECK-NEXT: icmp eq
; CHECK-NEXT: br i1 %cmp, label %exit2, label %exit1
; CHECK-NOT: icmp ne
entry:
  %cmp2 = icmp ne ptr %p, null
  %cmp = icmp eq ptr %p, null
  br i1 %cmp, label %is_null, label %not_null
is_null:
  br i1 %cmp2, label %exit1, label %exit2
not_null:
  br i1 %cmp2, label %exit1, label %exit2
exit1:
  ret i32 0
exit2:
  ret i32 1
}
