; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; This vscale udiv with a power-of-2 spalt on the rhs should not crash opt

; CHECK: define <vscale x 2 x i32> @udiv_pow2_vscale(<vscale x 2 x i32> %lhs)
define <vscale x 2 x i32> @udiv_pow2_vscale(<vscale x 2 x i32> %lhs) {
  %splatter = insertelement <vscale x 2 x i32> poison, i32 2, i32 0
  %rhs = shufflevector <vscale x 2 x i32> %splatter,
                       <vscale x 2 x i32> undef,
                       <vscale x 2 x i32> zeroinitializer
  %res = udiv <vscale x 2 x i32> %lhs, %rhs
  ret <vscale x 2 x i32> %res
}

; This fixed width udiv with a power-of-2 splat on the rhs should also not
; crash, and instcombine should eliminate the udiv

; CHECK-LABEL: define <2 x i32> @udiv_pow2_fixed(<2 x i32> %lhs)
; CHECK-NOT: udiv
define <2 x i32> @udiv_pow2_fixed(<2 x i32> %lhs) {
  %splatter = insertelement <2 x i32> poison, i32 2, i32 0
  %rhs = shufflevector <2 x i32> %splatter,
                       <2 x i32> undef,
                       <2 x i32> zeroinitializer
  %res = udiv <2 x i32> %lhs, %rhs
  ret <2 x i32> %res
}
