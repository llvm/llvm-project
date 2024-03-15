; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for rcp are generated for float, double, and half.

; CHECK-LABEL: rcp_float4
; CHECK: fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %{{.*}}
define noundef <4 x float> @rcp_float4(<4 x float> noundef %p0) {
entry:
  %p0.addr = alloca <4 x float>, align 16
  store <4 x float> %p0, ptr %p0.addr, align 16
  %0 = load <4 x float>, ptr %p0.addr, align 16
  %dx.rcp = call <4 x float> @llvm.dx.rcp.v4f32(<4 x float> %0)
  ret <4 x float> %dx.rcp
}

; CHECK-LABEL: rcp_double4
; CHECK: fdiv <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %{{.*}}
define noundef <4 x double> @rcp_double4(<4 x double> noundef %p0) {
entry:
  %p0.addr = alloca <4 x double>, align 16
  store <4 x double> %p0, ptr %p0.addr, align 16
  %0 = load <4 x double>, ptr %p0.addr, align 16
  %dx.rcp = call <4 x double> @llvm.dx.rcp.v4f64(<4 x double> %0)
  ret <4 x double> %dx.rcp
}

; CHECK-LABEL: rcp_half4
; CHECK: fdiv <4 x half> <half  0xH3C00, half  0xH3C00, half  0xH3C00, half  0xH3C00>, %{{.*}} 
define noundef <4 x half> @rcp_half4(<4 x half> noundef %p0) {
entry:
  %p0.addr = alloca <4 x half>, align 16
  store <4 x half> %p0, ptr %p0.addr, align 16
  %0 = load <4 x half>, ptr %p0.addr, align 16
  %dx.rcp = call <4 x half> @llvm.dx.rcp.v4f16(<4 x half> %0)
  ret <4 x half> %dx.rcp
}

; CHECK-LABEL: rcp_half
; CHECK: fdiv half 0xH3C00, %{{.*}} 
define noundef half @rcp_half(half noundef %p0) {
entry:
  %p0.addr = alloca half, align 2
  store half %p0, ptr %p0.addr, align 2
  %0 = load half, ptr %p0.addr, align 2
  %dx.rcp = call half @llvm.dx.rcp.f16(half %0)
  ret half %dx.rcp
}

declare half @llvm.dx.rcp.f16(half)
declare <4 x half> @llvm.dx.rcp.v4f16(<4 x half>)
declare <4 x float> @llvm.dx.rcp.v4f32(<4 x float>)
declare <4 x double> @llvm.dx.rcp.v4f64(<4 x double>)
