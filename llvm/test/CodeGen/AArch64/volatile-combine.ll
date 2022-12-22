; RUN: llc -mtriple=arm64-apple-ios %s -o - -stop-after=finalize-isel 2>&1 | FileCheck %s

define void @foo(i64 %a, i64 %b, ptr %ptr) {
; CHECK-LABEL: name: foo
; CHECK: STRWui {{.*}} (volatile store (s32) into %ir.ptr)
  %sum = add i64 %a, 1
  %sum.32 = trunc i64 %sum to i32
  store volatile i32 %sum.32, ptr %ptr
  ret void
}
