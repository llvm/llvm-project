; RUN: opt < %s -passes=dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; Declare a custom varargs function.
declare i16 @custom_varargs(i64, ...)

; CHECK-LABEL: @call_custom_varargs.dfsan
define void @call_custom_varargs(ptr %buf) {
  ;; All arguments have an annotation.  Check that the transformed function
  ;; preserves each annotation.

  ; CHECK: call zeroext i16 (i64, i8, ptr, ptr, ...)
  ; CHECK-SAME: @__dfsw_custom_varargs
  ; CHECK-SAME: i64 signext 200
  ; CHECK-SAME: ptr nonnull
  ; CHECK-SAME: i64 zeroext 20
  ; CHECK-SAME: i32 signext 1
  %call = call zeroext i16 (i64, ...) @custom_varargs(
    i64 signext 200,
    ptr nonnull %buf,
    i64 zeroext 20,
    i32 signext 1
  )
  ret void
}
