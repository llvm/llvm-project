; RUN: llc < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; Check that we emit the `@bar_*` symbols, and that we don't emit multiple symbols.

; CHECK-LABEL: .Lrel_0:
; CHECK: .long   foo_0@GOTPCREL+0
; CHECK-LABEL: .Lrel_1_failed:
; CHECK: .long   bar_1-foo_0
; CHECK-LABEL: .Lrel_2:
; CHECK: .long   foo_2@GOTPCREL+0

; CHECK: bar_0:
; CHECK: bar_1:
; CHECK: bar_2_indirect:

@rel_0 = private unnamed_addr constant [1 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @bar_0 to i64), i64 ptrtoint (ptr @rel_0 to i64)) to i32)]
@rel_1_failed = private unnamed_addr constant [1 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @bar_1 to i64), i64 ptrtoint (ptr @foo_0 to i64)) to i32)]
@rel_2 = private unnamed_addr constant [1 x i32] [
  i32 trunc (i64 sub (i64 ptrtoint (ptr @bar_2_indirect to i64), i64 ptrtoint (ptr @rel_2 to i64)) to i32)]
@bar_0 = internal unnamed_addr constant ptr @foo_0, align 8
@bar_1 = internal unnamed_addr constant ptr @foo_1, align 8
@bar_2_indirect = internal unnamed_addr constant ptr @foo_2, align 8
@foo_0 = external global ptr, align 8
@foo_1 = external global ptr, align 8
@foo_2 = external global ptr, align 8

define void @foo(ptr %arg0, ptr %arg1) {
  store ptr @bar_0, ptr %arg0, align 8
  store ptr @bar_1, ptr %arg1, align 8
  store ptr getelementptr (i8, ptr @bar_2_indirect, i32 1), ptr %arg1, align 8
  ret void
}
