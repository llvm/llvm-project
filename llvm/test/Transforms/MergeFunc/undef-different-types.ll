; RUN: opt -passes=mergefunc -S < %s | FileCheck %s
; RUN: opt -passes=mergefunc -S < %s | FileCheck -check-prefix=MERGE %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Cfunc and Dfunc differ only in that one returns i64, the other a pointer, and
; both return undef. They should be merged. Note undef cannot be merged with
; anything else, because this implies the ordering will be inconsistent (i.e.
; -1 == undef and undef == 1, but -1 < 1, so we must have undef != <any int>).
define internal i64 @Cfunc(ptr %P, ptr %Q) {
; CHECK-LABEL: define internal i64 @Cfunc
  store i32 4, ptr %P
  store i32 6, ptr %Q
  ret i64 undef
}

define internal ptr @Dfunc(ptr %P, ptr %Q) {
; MERGE-NOT: @Dfunc
  store i32 4, ptr %P
  store i32 6, ptr %Q
  ret ptr undef
}
