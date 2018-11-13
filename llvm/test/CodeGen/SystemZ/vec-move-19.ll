; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test that a loaded value which is replicated is not inserted also in any
; elements.

; CHECK:      vlvgp   %v0, %r0, %r0
; CHECK-NEXT: vrepf   %v24, %v0, 1
; CHECK-NOT:  vlvgf   %v24, %r0, 1
; CHECK-NOT:  vlvgf   %v24, %r0, 2

define <4 x i32> @fun(i32 %arg, i32* %dst) {
  %tmp = load i32, i32* undef
  %tmp8 = insertelement <4 x i32> undef, i32 %tmp, i32 0
  %tmp9 = insertelement <4 x i32> %tmp8, i32 %tmp, i32 1
  %tmp10 = insertelement <4 x i32> %tmp9, i32 %tmp, i32 2
  %tmp11 = insertelement <4 x i32> %tmp10, i32 %arg, i32 3
  store i32 %tmp, i32* %dst
  ret <4 x i32> %tmp11
}

