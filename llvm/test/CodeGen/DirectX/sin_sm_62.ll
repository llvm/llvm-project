; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s -check-prefix=SM6_3
; Half and float are valid for SM6.2 and later
; SM6_3: call half @dx.op.unary.f16(i32 13, half %{{.*}})
; SM6_3: call float @dx.op.unary.f32(i32 13, float %{{.*}})

; Function Attrs: noinline nounwind optnone
define noundef half @sin_half(half noundef %a) #0 {
entry:
  %a.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %1 = call half @llvm.sin.f16(half %0)
  ret half %1
}

; Function Attrs: noinline nounwind optnone
define noundef float @sin_float(float noundef %a) #0 {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = call float @llvm.sin.f32(float %0)
  ret float %1
}

