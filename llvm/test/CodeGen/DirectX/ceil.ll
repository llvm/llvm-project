; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for ceil are generated for float and double.

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"

; Function Attrs: noinline nounwind optnone
define noundef float @_Z3foof(float noundef %a) #0 {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  %0 = load float, ptr %a.addr, align 4
  ; CHECK:call float @dx.op.unary.f32(i32 28, float %{{.*}})
  %1 = call float @llvm.ceil.f32(float %0)
  ret float %1
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.ceil.f32(float) #1

; Function Attrs: noinline nounwind optnone
define noundef double @_Z3barDh(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  ; CHECK:call double @dx.op.unary.f64(i32 28, double %{{.*}})
  %1 = call double @llvm.ceil.f64(double %0)
  ret double %1
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.ceil.f64(double) #1

attributes #0 = { noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 73417c517644db5c419c85c0b3cb6750172fcab5)"}
