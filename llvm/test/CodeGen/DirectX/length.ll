; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S  -dxil-op-lower  < %s | FileCheck %s --check-prefixes=CHECK,DOPCHECK

; ModuleID = 'D:\llvm-project\clang\test\CodeGenHLSL\builtins\length.hlsl'
source_filename = "D:\\llvm-project\\clang\\test\\CodeGenHLSL\\builtins\\length.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.3-pc-shadermodel6.3-library"

; Function Attrs: convergent noinline nounwind optnone
define noundef float @"?test_length_half@@YA$halff@$halff@@Z"(float noundef %p0) #0 {
entry:
  %p0.addr = alloca float, align 4
  store float %p0, ptr %p0.addr, align 4
  %0 = load float, ptr %p0.addr, align 4

  ; EXPCHECK: call float @llvm.fabs.f32(float %{{.*}})
  ; DOPCHECK: call float @dx.op.unary.f32(i32 6, float %{{.*}})
  %1 = call float @llvm.fabs.f32(float %0) #3
  ret float %1
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #1

; Function Attrs: convergent noinline nounwind optnone
define noundef float @"?test_length_half2@@YA$halff@T?$__vector@$halff@$01@__clang@@@Z"(<2 x float> noundef %p0) #0 {
entry:
  %p0.addr = alloca <2 x float>, align 8
  store <2 x float> %p0, ptr %p0.addr, align 8
  %0 = load <2 x float>, ptr %p0.addr, align 8

  ; CHECK: extractelement <2 x float> %{{.*}}, i64 0
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <2 x float> %{{.*}}, i64 1
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; EXPCHECK: call float @llvm.sqrt.f32(float %{{.*}})
  ; DOPCHECK: call float @dx.op.unary.f32(i32 24, float %{{.*}})

  %hlsl.length = call float @llvm.dx.length.v2f32(<2 x float> %0)
  ret float %hlsl.length
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare float @llvm.dx.length.v2f32(<2 x float>) #2

; Function Attrs: convergent noinline nounwind optnone
define noundef float @"?test_length_half3@@YA$halff@T?$__vector@$halff@$02@__clang@@@Z"(<3 x float> noundef %p0) #0 {
entry:
  %p0.addr = alloca <3 x float>, align 16
  store <3 x float> %p0, ptr %p0.addr, align 16
  %0 = load <3 x float>, ptr %p0.addr, align 16

  ; CHECK: extractelement <3 x float> %{{.*}}, i64 0
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <3 x float> %{{.*}}, i64 1
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <3 x float> %{{.*}}, i64 2
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; EXPCHECK: call float @llvm.sqrt.f32(float %{{.*}})
  ; DOPCHECK: call float @dx.op.unary.f32(i32 24, float %{{.*}})

  %hlsl.length = call float @llvm.dx.length.v3f32(<3 x float> %0)
  ret float %hlsl.length
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare float @llvm.dx.length.v3f32(<3 x float>) #2

; Function Attrs: convergent noinline nounwind optnone
define noundef float @"?test_length_half4@@YA$halff@T?$__vector@$halff@$03@__clang@@@Z"(<4 x float> noundef %p0) #0 {
entry:
  %p0.addr = alloca <4 x float>, align 16
  store <4 x float> %p0, ptr %p0.addr, align 16
  %0 = load <4 x float>, ptr %p0.addr, align 16

  ; CHECK: extractelement <4 x float> %{{.*}}, i64 0
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <4 x float> %{{.*}}, i64 1
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <4 x float> %{{.*}}, i64 2
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <4 x float> %{{.*}}, i64 3
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; EXPCHECK: call float @llvm.sqrt.f32(float %{{.*}})
  ; DOPCHECK:  call float @dx.op.unary.f32(i32 24, float %{{.*}})

  %hlsl.length = call float @llvm.dx.length.v4f32(<4 x float> %0)
  ret float %hlsl.length
}

attributes #0 = { convergent noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }
attributes #3 = { memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{!"clang version 20.0.0git (git@github.com:bob80905/llvm-project.git 2fa4ffdc63e699e2b0e3c44e5dfb95284dbc5f6b)"}