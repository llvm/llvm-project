; RUN: opt -passes=mergefunc -S < %s | FileCheck %s
; RUN: opt -passes=mergefunc -S < %s | FileCheck -check-prefix=MERGE %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Afunc and Bfunc differ only in that one returns i64, the other a pointer.
; These should be merged.
define internal i64 @Afunc(ptr %P, ptr %Q) {
; CHECK-LABEL: define internal i64 @Afunc
  store i32 4, ptr %P
  store i32 6, ptr %Q
  ret i64 0
}

define internal ptr @Bfunc(ptr %P, ptr %Q) {
; MERGE-NOT: @Bfunc
  store i32 4, ptr %P
  store i32 6, ptr %Q
  ret ptr null
}

