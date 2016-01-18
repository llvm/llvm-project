; RUN: opt -S -instcombine < %s | FileCheck %s

; In PR20059 ( http://llvm.org/pr20059 ), shufflevector operations are reordered/removed
; for an srem operation. This is not a valid optimization because it may cause a trap
; on div-by-zero.

; CHECK-LABEL: @do_not_reorder
; CHECK: %splat1 = shufflevector <4 x i32> %p1, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT: %splat2 = shufflevector <4 x i32> %p2, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT: %retval = srem <4 x i32> %splat1, %splat2
define <4 x i32> @do_not_reorder(<4 x i32> %p1, <4 x i32> %p2) {
  %splat1 = shufflevector <4 x i32> %p1, <4 x i32> undef, <4 x i32> zeroinitializer
  %splat2 = shufflevector <4 x i32> %p2, <4 x i32> undef, <4 x i32> zeroinitializer
  %retval = srem <4 x i32> %splat1, %splat2
  ret <4 x i32> %retval
}
