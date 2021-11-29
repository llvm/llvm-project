; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@a = internal global i64* null, align 8
; CHECK: @a

; PR13968
define void @qux_no_null_opt() nounwind #0 {
; CHECK-LABEL: @qux_no_null_opt(
; CHECK: getelementptr i64*, i64** @a, i32 1
; CHECK: store i64* inttoptr (i64 1 to i64*), i64** @a
  %b = bitcast i64** @a to i8*
  %g = getelementptr i64*, i64** @a, i32 1
  %cmp = icmp ne i8* null, %b
  %cmp2 = icmp eq i8* null, %b
  %cmp3 = icmp eq i64** null, %g
  store i64* inttoptr (i64 1 to i64*), i64** @a, align 8
  %l = load i64*, i64** @a, align 8
  ret void
}

define i64* @bar() {
  %X = load i64*, i64** @a, align 8
  ret i64* %X
; CHECK-LABEL: @bar(
; CHECK: load
}

attributes #0 = { null_pointer_is_valid }
