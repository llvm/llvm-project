; Vector llvm.minimum / llvm.maximum lower to native AIR vector fmin/fmax
; (Apple has air.fmin.v2f32 / air.fmax.v2f32) with a NaN-propagation guard
; (fcmp uno + select), since air.fmin/fmax flush NaN.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: call <2 x float> @llvm.minimum.v2f32
; CHECK-NOT: call <2 x float> @llvm.maximum.v2f32
; CHECK: call <2 x float> @air.fmin.v2f32
; CHECK: fcmp uno <2 x float>
; CHECK: select <2 x i1> {{.*}}, <2 x float> splat (float +qnan)
; CHECK: call <2 x float> @air.fmax.v2f32
; CHECK: fcmp uno <2 x float>
; CHECK: select <2 x i1> {{.*}}, <2 x float> splat (float +qnan)
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
