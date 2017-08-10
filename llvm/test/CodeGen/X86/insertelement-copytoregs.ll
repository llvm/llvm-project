; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s
; CHECK-NOT: IMPLICIT_DEF

define void @foo(<2 x float>* %p) {
  %t = insertelement <2 x float> undef, float 0.0, i32 0
  %v = insertelement <2 x float> %t,   float 0.0, i32 1
  br label %bb8

bb8:
  store <2 x float> %v, <2 x float>* %p
  ret void
}
