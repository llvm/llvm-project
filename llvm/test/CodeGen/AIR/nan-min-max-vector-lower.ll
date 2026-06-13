; Positive coverage for vector llvm.minimum / llvm.maximum lowering to native
; AIR vector fmin/fmax + NaN-propagation guard across <2 x float>, <4 x float>,
; and <2 x half>.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

declare <2 x float> @llvm.minimum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.maximum.v4f32(<4 x float>, <4 x float>)
declare <2 x half>  @llvm.minimum.v2f16(<2 x half>,  <2 x half>)

; CHECK-LABEL: define <2 x float> @v2f32_min
; CHECK-NOT: call <2 x float> @llvm.minimum.v2f32
; CHECK: call <2 x float> @air.fmin.v2f32
; CHECK: fcmp uno <2 x float>
; CHECK: select <2 x i1> {{.*}}, <2 x float> splat (float +qnan)
define <2 x float> @v2f32_min(<2 x float> %a, <2 x float> %b) {
  %r = call <2 x float> @llvm.minimum.v2f32(<2 x float> %a, <2 x float> %b)
  ret <2 x float> %r
}

; CHECK-LABEL: define <4 x float> @v4f32_max
; CHECK-NOT: call <4 x float> @llvm.maximum.v4f32
; CHECK: call <4 x float> @air.fmax.v4f32
; CHECK: fcmp uno <4 x float>
; CHECK: select <4 x i1> {{.*}}, <4 x float> splat (float +qnan)
define <4 x float> @v4f32_max(<4 x float> %a, <4 x float> %b) {
  %r = call <4 x float> @llvm.maximum.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %r
}

; CHECK-LABEL: define <2 x half> @v2f16_min
; CHECK-NOT: call <2 x half> @llvm.minimum.v2f16
; CHECK: call <2 x half> @air.fmin.v2f16
; CHECK: fcmp uno <2 x half>
; CHECK: select <2 x i1> {{.*}}, <2 x half> splat (half +qnan)
define <2 x half> @v2f16_min(<2 x half> %a, <2 x half> %b) {
  %r = call <2 x half> @llvm.minimum.v2f16(<2 x half> %a, <2 x half> %b)
  ret <2 x half> %r
}
