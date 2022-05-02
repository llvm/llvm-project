; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for sin are generated for float and half.
; CHECK:call float @dx.op.unary.f32(i32 13, float %{{.*}})
; CHECK:call half @dx.op.unary.f16(i32 13, half %{{.*}})

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"

; Function Attrs: noinline nounwind optnone
define noundef float @_Z3foof(float noundef %a) #0 {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  %1 = call float @llvm.sin.f32(float %0)
  ret float %1
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.sin.f32(float) #1

; Function Attrs: noinline nounwind optnone
define noundef half @_Z3barDh(half noundef %a) #0 {
entry:
  %a.addr = alloca half, align 2
  store half %a, ptr %a.addr, align 2
  %0 = load half, ptr %a.addr, align 2
  %1 = call half @llvm.sin.f16(half %0)
  ret half %1
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare half @llvm.sin.f16(half) #1

attributes #0 = { noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 73417c517644db5c419c85c0b3cb6750172fcab5)"}
