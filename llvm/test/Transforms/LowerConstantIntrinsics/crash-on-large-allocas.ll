; RUN: opt -S -lower-constant-intrinsics %s -o - | FileCheck %s
;
; Ensure that we don't {crash,return a bad value} when given an alloca larger
; than what a pointer can represent.

target datalayout = "p:16:16"

; CHECK-LABEL: @alloca_overflow_is_unknown(
define i16 @alloca_overflow_is_unknown() {
  %i = alloca i8, i32 65537
  %j = call i16 @llvm.objectsize.i16.p0(ptr %i, i1 false, i1 false, i1 false)
  ; CHECK: ret i16 -1
  ret i16 %j
}

declare i16 @llvm.objectsize.i16.p0(ptr, i1, i1, i1)
