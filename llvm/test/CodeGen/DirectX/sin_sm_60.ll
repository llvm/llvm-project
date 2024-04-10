; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library %s | FileCheck %s -check-prefix=SM6_0_FLOAT

; Float is valid for SM6.0
; SM6_0_FLOAT: call float @dx.op.unary.f32(i32 13, float %{{.*}})

; Function Attrs: noinline nounwind optnone
define noundef float @sin_float(float noundef %a) #0 {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = call float @llvm.sin.f32(float %0)
  ret float %1
}
