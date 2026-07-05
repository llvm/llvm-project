; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 | FileCheck %s

; Ensure this does not crash

define <2 x i8> @test1(<2 x i8> %a) {
  %1 = shl nuw <2 x i8> %a, <i8 7, i8 7>
  %2 = ashr exact <2 x i8> %1, <i8 7, i8 7>
  ret <2 x i8> %2
}
; CHECK-LABEL: @test1
; CHECK: vspltisb [[REG1:[0-9]+]], 7
; CHECK: vslb [[REG2:[0-9]+]], 2, [[REG1]]
; CHECK: vsrab [[REG3:[0-9]+]], [[REG2]], [[REG1]]

define <2 x i16> @test2(<2 x i16> %a) {
  %1 = shl nuw <2 x i16> %a, <i16 15, i16 15>
  %2 = ashr exact <2 x i16> %1, <i16 15, i16 15>
  ret <2 x i16> %2
}

; CHECK-LABEL: @test2
; CHECK: vspltish [[REG1:[0-9]+]], 15
; CHECK: vslh [[REG2:[0-9]+]], 2, [[REG1]]
; CHECK: vsrah [[REG3:[0-9]+]], [[REG2]], [[REG1]]
