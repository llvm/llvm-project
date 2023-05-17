; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@a = internal global ptr null, align 8
; CHECK: @a

; PR13968
define void @qux_no_null_opt() nounwind #0 {
; CHECK-LABEL: @qux_no_null_opt(
; CHECK: getelementptr ptr, ptr @a, i32 1
; CHECK: store ptr inttoptr (i64 1 to ptr), ptr @a
  %g = getelementptr ptr, ptr @a, i32 1
  %cmp = icmp ne ptr null, @a
  %cmp2 = icmp eq ptr null, @a
  %cmp3 = icmp eq ptr null, %g
  store ptr inttoptr (i64 1 to ptr), ptr @a, align 8
  %l = load ptr, ptr @a, align 8
  ret void
}

define ptr @bar() {
  %X = load ptr, ptr @a, align 8
  ret ptr %X
; CHECK-LABEL: @bar(
; CHECK: load
}

attributes #0 = { null_pointer_is_valid }
