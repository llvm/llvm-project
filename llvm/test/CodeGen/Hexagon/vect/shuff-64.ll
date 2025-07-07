; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_00:
; CHECK-DAG: r[[REG00:[0-9]+]] = swiz(r0)
; CHECK-DAG: r[[REG01:[0-9]+]] = swiz(r1)
; CHECK: r1:0 = combine(r[[REG00]],r[[REG01]])
define <8 x i8> @test_00(<8 x i8> %a0) {
  %p = shufflevector <8 x i8> %a0, <8 x i8> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i8> %p
}

; CHECK-LABEL: test_10:
; CHECK: r1:0 = packhl(r1,r0)
define <4 x i16> @test_10(<4 x i16> %a0) {
  %p = shufflevector <4 x i16> %a0, <4 x i16> undef, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ret <4 x i16> %p
}

; CHECK-LABEL: test_11:
; CHECK: r1:0 = packhl(r1,r0)
define <4 x i16> @test_11(<4 x i16> %a0) {
  %p = shufflevector <4 x i16> undef, <4 x i16> %a0, <4 x i32> <i32 4, i32 6, i32 5, i32 7>
  ret <4 x i16> %p
}

; CHECK-LABEL: test_20:
; CHECK: r1:0 = shuffeh(r3:2,r1:0)
define <4 x i16> @test_20(<4 x i16> %a0, <4 x i16> %a1) {
  %p = shufflevector <4 x i16> %a0, <4 x i16> %a1, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x i16> %p
}

; CHECK-LABEL: test_30:
; CHECK: r1:0 = shuffoh(r3:2,r1:0)
define <4 x i16> @test_30(<4 x i16> %a0, <4 x i16> %a1) {
  %p = shufflevector <4 x i16> %a0, <4 x i16> %a1, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x i16> %p
}

; CHECK-LABEL: test_40:
; CHECK: r1:0 = vtrunewh(r3:2,r1:0)
define <4 x i16> @test_40(<4 x i16> %a0, <4 x i16> %a1) {
  %p = shufflevector <4 x i16> %a0, <4 x i16> %a1, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret <4 x i16> %p
}

; CHECK-LABEL: test_50:
; CHECK: r1:0 = vtrunowh(r3:2,r1:0)
define <4 x i16> @test_50(<4 x i16> %a0, <4 x i16> %a1) {
  %p = shufflevector <4 x i16> %a0, <4 x i16> %a1, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x i16> %p
}

; CHECK-LABEL: test_60:
; r1:0 = shuffeb(r3:2,r1:0)
define <8 x i8> @test_60(<8 x i8> %a0, <8 x i8> %a1) {
  %p = shufflevector <8 x i8> %a0, <8 x i8> %a1, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i8> %p
}

; CHECK-LABEL: test_70:
; r1:0 = shuffob(r3:2,r1:0)
define <8 x i8> @test_70(<8 x i8> %a0, <8 x i8> %a1) {
  %p = shufflevector <8 x i8> %a0, <8 x i8> %a1, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i8> %p
}
