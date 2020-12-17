; RUN: opt %s -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 | FileCheck %s
; Check for when one branch implies the value of a successors conditional and
; it's not simply the same conditional repeated.

define void @test(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test
  %iplus1 = add nsw i32 %i, 1
  %var29 = icmp slt i32 %iplus1, %length.i
; CHECK: br i1 %var29, label %in_bounds, label %out_of_bounds
  br i1 %var29, label %next, label %out_of_bounds

next:
; CHECK-LABEL: in_bounds:
; CHECK-NEXT: ret void
  %var30 = icmp slt i32 %i, %length.i
  br i1 %var30, label %in_bounds, label %out_of_bounds2

in_bounds:
  ret void

out_of_bounds:
  call void @foo(i64 0)
  unreachable

out_of_bounds2:
  call void @foo(i64 1)
  unreachable
}

; If the add is not nsw, it's not safe to use the fact about i+1 to imply the
; i condition since it could have overflowed.
define void @test_neg(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test_neg
  %iplus1 = add i32 %i, 1
  %var29 = icmp slt i32 %iplus1, %length.i
; CHECK: br i1 %var29, label %next, label %out_of_bounds
  br i1 %var29, label %next, label %out_of_bounds

next:
  %var30 = icmp slt i32 %i, %length.i
; CHECK: br i1 %var30, label %in_bounds, label %out_of_bounds2
  br i1 %var30, label %in_bounds, label %out_of_bounds2

in_bounds:
  ret void

out_of_bounds:
  call void @foo(i64 0)
  unreachable

out_of_bounds2:
  call void @foo(i64 1)
  unreachable
}


define void @test2(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test2
  %iplus100 = add nsw i32 %i, 100
  %var29 = icmp slt i32 %iplus100, %length.i
; CHECK: br i1 %var29, label %in_bounds, label %out_of_bounds
  br i1 %var29, label %next, label %out_of_bounds

next:
  %var30 = icmp slt i32 %i, %length.i
  br i1 %var30, label %in_bounds, label %out_of_bounds2

in_bounds:
  ret void

out_of_bounds:
  call void @foo(i64 0)
  unreachable

out_of_bounds2:
  call void @foo(i64 1)
  unreachable
}

declare void @foo(i64)

