; RUN: llc -mtriple=aarch64-none-linux-gnu < %s -mattr=+neon | FileCheck %s

define <4 x i32> @test1(<4 x i32> %a) {
  %rem = srem <4 x i32> %a, <i32 7, i32 7, i32 7, i32 7>
  ret <4 x i32> %rem
; CHECK-LABEL: test1
; FIXME: Can we lower this more efficiently?
; CHECK: mul
; CHECK: mul
; CHECK: mul
; CHECK: mul
}

