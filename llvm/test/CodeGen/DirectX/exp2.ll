; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for exp2 are generated for float and half.
; CHECK:call float @dx.op.unary.f32(i32 21, float %{{.*}})
; CHECK:call half @dx.op.unary.f16(i32 21, half %{{.*}})

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"

; Function Attrs: noinline nounwind optnone
define noundef float @exp2_float(float noundef %a) #0 {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %elt.exp2 = call float @llvm.exp2.f32(float %0)
  ret float %elt.exp2
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.exp2.f32(float) #1

; Function Attrs: noinline nounwind optnone
define noundef half @exp2_half(half noundef %a) #0 {
entry:
  %a.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %elt.exp2 = call half @llvm.exp2.f16(half %0)
  ret half %elt.exp2
}
