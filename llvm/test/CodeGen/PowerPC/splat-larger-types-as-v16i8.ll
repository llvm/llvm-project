; RUN: llc -mcpu=pwr9 -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s
define <8 x i16> @test1() {
entry:
  ret <8 x i16> <i16 257, i16 257, i16 257, i16 257, i16 257, i16 257, i16 257, i16 257>
; CHECK-LABEL: test1
; CHECK: xxspltib 34, 1
}
define <8 x i16> @testAB() {
entry:
; CHECK-LABEL: testAB
; CHECK: xxspltib 34, 171
  ret <8 x i16> <i16 43947, i16 43947, i16 43947, i16 43947, i16 43947, i16 43947, i16 43947, i16 43947>
}
define <4 x i32> @testAB32() {
entry:
; CHECK-LABEL: testAB32
; CHECK: xxspltib 34, 171
  ret <4 x i32> <i32 2880154539, i32 2880154539, i32 2880154539, i32 2880154539>
}
