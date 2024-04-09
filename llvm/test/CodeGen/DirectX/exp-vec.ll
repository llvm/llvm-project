; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s

; Make sure dxil operation function calls for exp are generated for float and half.

; CHECK-LABEL: exp_float4
; CHECK: fmul <4 x float> <float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000>,  %{{.*}}
; CHECK: call <4 x float> @llvm.exp2.v4f32(<4 x float>  %{{.*}})
define noundef <4 x float> @exp_float4(<4 x float> noundef %p0) {
entry:
  %p0.addr = alloca <4 x float>, align 16
  store <4 x float> %p0, ptr %p0.addr, align 16
  %0 = load <4 x float>, ptr %p0.addr, align 16
  %elt.exp = call <4 x float> @llvm.exp.v4f32(<4 x float> %0)
  ret <4 x float> %elt.exp
}

declare <4 x float> @llvm.exp.v4f32(<4 x float>)
