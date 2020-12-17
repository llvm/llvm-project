; REQUIRES: asserts
; RUN: not --crash llc %s -march=ve -mattr=+vpu -o /dev/null |& FileCheck %s

; CHECK:  t{{[0-9]+}}: v256i32 = vp_udiv [[A:t[0-9]+]], [[B:t[0-9]+]], [[MASK:t[0-9]+]], [[EVL:t[0-9]+]] 
; CHECK:  [[A]]: v256i32
; CHECK:  [[B]]: v256i32
; CHECK:  [[MASK]]: v256i1
; CHECK:  [[EVL]]: i32

define <256 x i32> @test_vp_int(<256 x i32> %i0, <256 x i32> %i1, <256 x i1> %m, i32 %n) {
  %r0 = call <256 x i32> @llvm.vp.udiv.v256i32(<256 x i32> %i0, <256 x i32> %i1, <256 x i1> %m, i32 %n)
  ret <256 x i32> %r0
}

; integer arith
declare <256 x i32> @llvm.vp.udiv.v256i32(<256 x i32>, <256 x i32>, <256 x i1>, i32)
