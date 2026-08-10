; RUN: llc < %s -mtriple=sparc | FileCheck %s

; Verify that we correctly handle vector types that appear directly
; during call lowering. These may cause issue as v2i32 is a legal type
; for the implementation of LDD

; CHECK-LABEL: fun16v:
; CHECK: foo1_16v
; CHECK: foo2_16v

define <2 x i16> @fun16v() #0 {
  %1 = tail call <2 x i16> @foo1_16v()
  %2 = tail call <2 x i16> @foo2_16v()
  %3 = and <2 x i16> %2, %1
  ret <2 x i16> %3
}

declare <2 x i16> @foo1_16v() #0
declare <2 x i16> @foo2_16v() #0

; CHECK-LABEL: fun32v:
; CHECK: foo1_32v
; CHECK: foo2_32v

define <2 x i32> @fun32v() #0 {
  %1 = tail call <2 x i32> @foo1_32v()
  %2 = tail call <2 x i32> @foo2_32v()
  %3 = and <2 x i32> %2, %1
  ret <2 x i32> %3
}

declare <2 x i32> @foo1_32v() #0
declare <2 x i32> @foo2_32v() #0
