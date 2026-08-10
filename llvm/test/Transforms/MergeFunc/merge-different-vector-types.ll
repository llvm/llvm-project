; RUN: opt -passes=mergefunc -S < %s | FileCheck %s
; RUN: opt -passes=mergefunc -S < %s | FileCheck -check-prefix=MERGE %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Merging should still work even if the values are wrapped in a vector.
define internal <2 x i64> @Mfunc(ptr %P, ptr %Q) {
; CHECK-LABEL: define internal <2 x i64> @Mfunc
  store i32 1, ptr %P
  store i32 1, ptr %Q
  ret <2 x i64> <i64 0, i64 0>
}

define internal <2 x ptr> @Nfunc(ptr %P, ptr %Q) {
; MERGE-NOT: @Nfunc
  store i32 1, ptr %P
  store i32 1, ptr %Q
  ret <2 x ptr> <ptr null, ptr null>
}
