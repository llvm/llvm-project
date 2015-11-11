; RUN: llc -mcpu=pwr8 -mattr=+vsx -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

define <2 x double> @test00(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 0, i32 0>
  ret <2 x double> %v3

; CHECK-LABEL: test00
; CHECK: lxvd2x 0, 0, 3
; CHECK: xxspltd 34, 0, 0
}

define <2 x double> @test01(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 0, i32 1>
  ret <2 x double> %v3

; CHECK-LABEL: test01
; CHECK: lxvd2x 0, 0, 3
; CHECK: xxswapd 34, 0
}

define <2 x double> @test02(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 0, i32 2>
  ret <2 x double> %v3

; CHECK-LABEL: @test02
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxvd2x 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxswapd 1, 1
; CHECK: xxmrgld 34, 1, 0
}

define <2 x double> @test03(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 0, i32 3>
  ret <2 x double> %v3

; CHECK-LABEL: @test03
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxvd2x 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxswapd 1, 1
; CHECK: xxpermdi 34, 1, 0, 1
}

define <2 x double> @test10(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 1, i32 0>
  ret <2 x double> %v3

; CHECK-LABEL: @test10
; CHECK: lxvd2x 34, 0, 3
}

define <2 x double> @test11(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 1, i32 1>
  ret <2 x double> %v3

; CHECK-LABEL: @test11
; CHECK: lxvd2x 0, 0, 3
; CHECK: xxspltd 34, 0, 1
}

define <2 x double> @test12(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 1, i32 2>
  ret <2 x double> %v3

; CHECK-LABEL: @test12
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxvd2x 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxswapd 1, 1
; CHECK: xxpermdi 34, 1, 0, 2
}

define <2 x double> @test13(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 1, i32 3>
  ret <2 x double> %v3

; CHECK-LABEL: @test13
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxvd2x 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxswapd 1, 1
; CHECK: xxmrghd 34, 1, 0
}

define <2 x double> @test20(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 2, i32 0>
  ret <2 x double> %v3

; CHECK-LABEL: @test20
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxvd2x 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxswapd 1, 1
; CHECK: xxmrgld 34, 0, 1
}

define <2 x double> @test21(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 2, i32 1>
  ret <2 x double> %v3

; CHECK-LABEL: @test21
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxvd2x 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxswapd 1, 1
; CHECK: xxpermdi 34, 0, 1, 1
}

define <2 x double> @test22(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 2, i32 2>
  ret <2 x double> %v3

; CHECK-LABEL: @test22
; CHECK: lxvd2x 0, 0, 4
; CHECK: xxspltd 34, 0, 0
}

define <2 x double> @test23(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 2, i32 3>
  ret <2 x double> %v3

; CHECK-LABEL: @test23
; CHECK: lxvd2x 0, 0, 4
; CHECK: xxswapd 34, 0
}

define <2 x double> @test30(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 3, i32 0>
  ret <2 x double> %v3

; CHECK-LABEL: @test30
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxvd2x 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxswapd 1, 1
; CHECK: xxpermdi 34, 0, 1, 2
}

define <2 x double> @test31(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 3, i32 1>
  ret <2 x double> %v3

; CHECK-LABEL: @test31
; CHECK: lxvd2x 0, 0, 3
; CHECK: lxvd2x 1, 0, 4
; CHECK: xxswapd 0, 0
; CHECK: xxswapd 1, 1
; CHECK: xxmrghd 34, 0, 1
}

define <2 x double> @test32(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 3, i32 2>
  ret <2 x double> %v3

; CHECK-LABEL: @test32
; CHECK: lxvd2x 34, 0, 4
}

define <2 x double> @test33(<2 x double>* %p1, <2 x double>* %p2) {
  %v1 = load <2 x double>, <2 x double>* %p1
  %v2 = load <2 x double>, <2 x double>* %p2
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <2 x i32> < i32 3, i32 3>
  ret <2 x double> %v3

; CHECK-LABEL: @test33
; CHECK: lxvd2x 0, 0, 4
; CHECK: xxspltd 34, 0, 1
}
