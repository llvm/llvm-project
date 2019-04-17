; RUN: opt -mergefunc -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; These should not be merged, as the datalayout says a pointer is 64 bits. No
; sext/zext is specified, so these functions could lower differently.
define internal i32 @Ffunc(i32* %P, i32* %Q) {
; CHECK-LABEL: define internal i32 @Ffunc
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: ret
  store i32 1, i32* %P
  store i32 3, i32* %Q
  ret i32 0
}

define internal i64* @Gfunc(i32* %P, i32* %Q) {
; CHECK-LABEL: define internal i64* @Gfunc
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: ret
  store i32 1, i32* %P
  store i32 3, i32* %Q
  ret i64* null
}
