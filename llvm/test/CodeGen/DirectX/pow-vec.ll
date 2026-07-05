; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s

; Make sure dxil operation function calls for pow are generated for float and half.

; CHECK-LABEL: pow_float4
; CHECK: call <4 x float> @llvm.log2.v4f32(<4 x float>  %a)
; CHECK: fmul <4 x float> %{{.*}}, %b
; CHECK: call <4 x float> @llvm.exp2.v4f32(<4 x float>  %{{.*}})
define noundef <4 x float> @pow_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
  %elt.pow = call <4 x float> @llvm.pow.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %elt.pow
}

declare <4 x float> @llvm.pow.v4f32(<4 x float>,<4 x float>)
