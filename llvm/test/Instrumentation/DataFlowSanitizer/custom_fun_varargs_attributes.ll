; RUN: opt < %s -dfsan -dfsan-args-abi -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
; RUN: opt < %s -dfsan                 -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

target triple = "x86_64-unknown-linux-gnu"

; Declare a custom varargs function.
declare i16 @custom_varargs(i64, ...)

; CHECK-LABEL: @"dfs$call_custom_varargs"
define void @call_custom_varargs(i8* %buf) {
  ;; All arguments have an annotation.  Check that the transformed function
  ;; preserves each annotation.

  ; CHECK: call zeroext i16 (i64, i[[#SBITS]], i[[#SBITS]]*, i[[#SBITS]]*, ...)
  ; CHECK-SAME: @__dfsw_custom_varargs
  ; CHECK-SAME: i64 signext 200
  ; CHECK-SAME: i8* nonnull
  ; CHECK-SAME: i64 zeroext 20
  ; CHECK-SAME: i32 signext 1
  %call = call zeroext i16 (i64, ...) @custom_varargs(
    i64 signext 200,
    i8* nonnull %buf,
    i64 zeroext 20,
    i32 signext 1
  )
  ret void
}
