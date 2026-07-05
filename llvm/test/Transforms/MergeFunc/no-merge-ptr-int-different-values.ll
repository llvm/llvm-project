; RUN: opt -passes=mergefunc -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; These should not be merged, as 1 != 0.
define internal i64 @Ifunc(ptr %P, ptr %Q) {
; CHECK-LABEL: define internal i64 @Ifunc
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: ret
  store i32 10, ptr %P
  store i32 10, ptr %Q
  ret i64 1
}

define internal ptr @Jfunc(ptr %P, ptr %Q) {
; CHECK-LABEL: define internal ptr @Jfunc
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: ret
  store i32 10, ptr %P
  store i32 10, ptr %Q
  ret ptr null
}
