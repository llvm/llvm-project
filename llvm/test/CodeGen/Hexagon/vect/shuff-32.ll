; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_00:
; CHECK: r0 = swiz(r0)
define <4 x i8> @test_00(<4 x i8> %a0) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_01:
; CHECK: r0 = swiz(r0)
define <4 x i8> @test_01(<4 x i8> %a0) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> undef, <4 x i32> <i32 undef, i32 2, i32 1, i32 0>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_02:
; CHECK: r0 = swiz(r0)
define <4 x i8> @test_02(<4 x i8> %a0) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> undef, <4 x i32> <i32 3, i32 undef, i32 1, i32 0>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_03:
; CHECK: r0 = swiz(r0)
define <4 x i8> @test_03(<4 x i8> %a0) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> undef, <4 x i32> <i32 3, i32 2, i32 undef, i32 undef>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_10:
; CHECK: r0 = vtrunehb(r1:0)
define <4 x i8> @test_10(<4 x i8> %a0, <4 x i8> %a1) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> %a1, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_11:
; CHECK: r0 = vtrunehb(r1:0)
define <4 x i8> @test_11(<4 x i8> %a0, <4 x i8> %a1) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> %a1, <4 x i32> <i32 undef, i32 2, i32 4, i32 undef>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_12:
; CHECK: r0 = vtrunehb(r1:0)
define <4 x i8> @test_12(<4 x i8> %a0, <4 x i8> %a1) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> %a1, <4 x i32> <i32 0, i32 undef, i32 4, i32 6>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_13:
; CHECK: r0 = vtrunehb(r1:0)
define <4 x i8> @test_13(<4 x i8> %a0, <4 x i8> %a1) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> %a1, <4 x i32> <i32 0, i32 2, i32 undef, i32 undef>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_20:
; CHECK: r0 = vtrunohb(r1:0)
define <4 x i8> @test_20(<4 x i8> %a0, <4 x i8> %a1) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> %a1, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_21:
; CHECK: r0 = vtrunohb(r1:0)
define <4 x i8> @test_21(<4 x i8> %a0, <4 x i8> %a1) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> %a1, <4 x i32> <i32 undef, i32 3, i32 5, i32 7>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_22:
; CHECK: r0 = vtrunohb(r1:0)
define <4 x i8> @test_22(<4 x i8> %a0, <4 x i8> %a1) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> %a1, <4 x i32> <i32 undef, i32 undef, i32 5, i32 7>
  ret <4 x i8> %p
}

; CHECK-LABEL: test_23:
; CHECK: r0 = vtrunohb(r1:0)
define <4 x i8> @test_23(<4 x i8> %a0, <4 x i8> %a1) {
  %p = shufflevector <4 x i8> %a0, <4 x i8> %a1, <4 x i32> <i32 1, i32 3, i32 5, i32 undef>
  ret <4 x i8> %p
}

