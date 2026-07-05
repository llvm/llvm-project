; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library < %s | FileCheck %s

; Make sure dxil operation function calls for lerp are generated for float and half.

; CHECK-LABEL: lerp_half
; CHECK: fsub half %{{.*}}, %{{.*}}
; CHECK: fmul half %{{.*}}, %{{.*}}
; CHECK: fadd half %{{.*}}, %{{.*}}
define noundef half @lerp_half(half noundef %p0) {
entry:
  %p0.addr = alloca half, align 2
  store half %p0, ptr %p0.addr, align 2
  %0 = load half, ptr %p0.addr, align 2
  %1 = load half, ptr %p0.addr, align 2
  %2 = load half, ptr %p0.addr, align 2
  %dx.lerp = call half @llvm.dx.lerp.f16(half %0, half %1, half %2)
  ret half %dx.lerp
}

; CHECK-LABEL: lerp_float
; CHECK: fsub float %{{.*}}, %{{.*}}
; CHECK: fmul float %{{.*}}, %{{.*}}
; CHECK: fadd float %{{.*}}, %{{.*}}
define noundef float @lerp_float(float noundef %p0, float noundef %p1) {
entry:
  %p1.addr = alloca float, align 4
  %p0.addr = alloca float, align 4
  store float %p1, ptr %p1.addr, align 4
  store float %p0, ptr %p0.addr, align 4
  %0 = load float, ptr %p0.addr, align 4
  %1 = load float, ptr %p0.addr, align 4
  %2 = load float, ptr %p0.addr, align 4
  %dx.lerp = call float @llvm.dx.lerp.f32(float %0, float %1, float %2)
  ret float %dx.lerp
}

; CHECK-LABEL: lerp_float4
; CHECK: fsub <4 x float> %{{.*}}, %{{.*}}
; CHECK: fmul <4 x float> %{{.*}}, %{{.*}}
; CHECK: fadd <4 x float> %{{.*}}, %{{.*}}
define noundef <4 x float> @lerp_float4(<4 x float> noundef %p0, <4 x float> noundef %p1) {
entry:
  %p1.addr = alloca <4 x float>, align 16
  %p0.addr = alloca <4 x float>, align 16
  store <4 x float> %p1, ptr %p1.addr, align 16
  store <4 x float> %p0, ptr %p0.addr, align 16
  %0 = load <4 x float>, ptr %p0.addr, align 16
  %1 = load <4 x float>, ptr %p0.addr, align 16
  %2 = load <4 x float>, ptr %p0.addr, align 16
  %dx.lerp = call <4 x float> @llvm.dx.lerp.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %2)
  ret <4 x float> %dx.lerp
}

declare half @llvm.dx.lerp.f16(half, half, half)
declare float @llvm.dx.lerp.f32(float, float, float)
declare <4 x float> @llvm.dx.lerp.v4f32(<4 x float>, <4 x float>, <4 x float>)
