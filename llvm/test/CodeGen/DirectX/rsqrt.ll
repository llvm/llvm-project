; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for rsqrt are generated for float and half.

; CHECK-LABEL: rsqrt_float
; CHECK: call float @dx.op.unary.f32(i32 25, float %{{.*}})
define noundef float @rsqrt_float(float noundef %a) {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %dx.rsqrt = call float @llvm.dx.rsqrt.f32(float %0)
  ret float %dx.rsqrt
}

; CHECK-LABEL: rsqrt_half
; CHECK: call half @dx.op.unary.f16(i32 25, half %{{.*}})
define noundef half @rsqrt_half(half noundef %a) {
entry:
  %a.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %dx.rsqrt = call half @llvm.dx.rsqrt.f16(half %0)
  ret half %dx.rsqrt
}

declare half @llvm.dx.rsqrt.f16(half)
declare float @llvm.dx.rsqrt.f32(float)
