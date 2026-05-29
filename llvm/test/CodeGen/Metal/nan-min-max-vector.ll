; Vector llvm.minimum / llvm.maximum are scalarized to per-lane scalar
; calls so the existing scalar rename + NaN-propagation select can handle
; them. AIR has no vector fmin/fmax intrinsic.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: call <2 x float> @llvm.minimum.v2f32
; CHECK-NOT: call <2 x float> @llvm.maximum.v2f32
; CHECK-NOT: call <2 x float> @air.fmin.v2f32
; CHECK: call float @air.fmin.f32
; CHECK: call float @air.fmin.f32
; CHECK: call float @air.fmax.f32
; CHECK: call float @air.fmax.f32
; CHECK: ret void

declare <2 x float> @llvm.minimum.v2f32(<2 x float>, <2 x float>)
declare <2 x float> @llvm.maximum.v2f32(<2 x float>, <2 x float>)

define void @kernel(<2 x float> %a, <2 x float> %b, ptr addrspace(1) %out) {
entry:
  %mn = call <2 x float> @llvm.minimum.v2f32(<2 x float> %a, <2 x float> %b)
  %mx = call <2 x float> @llvm.maximum.v2f32(<2 x float> %a, <2 x float> %b)
  %s  = fadd <2 x float> %mn, %mx
  store <2 x float> %s, ptr addrspace(1) %out
  ret void
}
