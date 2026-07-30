; RUN: llc -mtriple=arm64-unknown-unknown -o - %s

; Make sure we don't run into an assert in the aarch64 code selection when
; DAGCombining fails.

declare void @t()

define void @foo() {
  %c = bitcast i64 270458 to i64
  %t0 = lshr i64 %c, 422383
  %t1 = trunc i64 %t0 to i1
  br i1 %t1, label %BB1, label %BB0

BB0:
  call void @t()
  br label %BB1

BB1:
  ret void
}
