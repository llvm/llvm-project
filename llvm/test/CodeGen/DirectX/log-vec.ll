; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s

; Make sure dxil operation function calls for log are generated for float and half.

; CHECK-LABEL: log_float4
; CHECK: call <4 x float> @llvm.log2.v4f32(<4 x float>  %{{.*}})
; CHECK: fmul <4 x float> <float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000, float 0x3FE62E4300000000>,  %{{.*}}
define noundef <4 x float> @log_float4(<4 x float> noundef %p0) {
entry:
  %p0.addr = alloca <4 x float>, align 16
  store <4 x float> %p0, ptr %p0.addr, align 16
  %0 = load <4 x float>, ptr %p0.addr, align 16
  %elt.log = call <4 x float> @llvm.log.v4f32(<4 x float> %0)
  ret <4 x float> %elt.log
}

; CHECK-LABEL: log10_float4
; CHECK: call <4 x float> @llvm.log2.v4f32(<4 x float>  %{{.*}})
; CHECK: fmul <4 x float> <float 0x3FD3441340000000, float 0x3FD3441340000000, float 0x3FD3441340000000, float 0x3FD3441340000000>,  %{{.*}}
define noundef <4 x float> @log10_float4(<4 x float> noundef %p0) {
entry:
  %p0.addr = alloca <4 x float>, align 16
  store <4 x float> %p0, ptr %p0.addr, align 16
  %0 = load <4 x float>, ptr %p0.addr, align 16
  %elt.log10 = call <4 x float> @llvm.log10.v4f32(<4 x float> %0)
  ret <4 x float> %elt.log10
}

declare <4 x float> @llvm.log.v4f32(<4 x float>)
declare <4 x float> @llvm.log10.v4f32(<4 x float>)
