; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; ModuleID = '../clang/test/CodeGenHLSL/builtins/asuint-splitdouble.hlsl'
source_filename = "../clang/test/CodeGenHLSL/builtins/asuint-splitdouble.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.3-pc-shadermodel6.3-library"

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define noundef float @"?test_scalar@@YAMN@Z"(double noundef %D) local_unnamed_addr #0 {
entry:
  ; CHECK: [[CALL:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double %D)
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  %hlsl.asuint = tail call { i32, i32 } @llvm.dx.splitdouble.i32(double %D)
  %0 = extractvalue { i32, i32 } %hlsl.asuint, 0
  %1 = extractvalue { i32, i32 } %hlsl.asuint, 1
  %add = add i32 %0, %1
  %conv = uitofp i32 %add to float
  ret float %conv
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare { i32, i32 } @llvm.dx.splitdouble.i32(double) #1

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define noundef <3 x float> @"?test_vector@@YAT?$__vector@M$02@__clang@@T?$__vector@N$02@2@@Z"(<3 x double> noundef %D) local_unnamed_addr #0 {
entry:
  %0 = extractelement <3 x double> %D, i64 0
  ; CHECK-COUNT-3: [[CALL:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double {{.*}})
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  %hlsl.asuint = tail call { i32, i32 } @llvm.dx.splitdouble.i32(double %0)
  %1 = extractvalue { i32, i32 } %hlsl.asuint, 0
  %2 = extractvalue { i32, i32 } %hlsl.asuint, 1
  %3 = insertelement <3 x i32> poison, i32 %1, i64 0
  %4 = insertelement <3 x i32> poison, i32 %2, i64 0
  %5 = extractelement <3 x double> %D, i64 1
  %hlsl.asuint2 = tail call { i32, i32 } @llvm.dx.splitdouble.i32(double %5)
  %6 = extractvalue { i32, i32 } %hlsl.asuint2, 0
  %7 = extractvalue { i32, i32 } %hlsl.asuint2, 1
  %8 = insertelement <3 x i32> %3, i32 %6, i64 1
  %9 = insertelement <3 x i32> %4, i32 %7, i64 1
  %10 = extractelement <3 x double> %D, i64 2
  %hlsl.asuint3 = tail call { i32, i32 } @llvm.dx.splitdouble.i32(double %10)
  %11 = extractvalue { i32, i32 } %hlsl.asuint3, 0
  %12 = extractvalue { i32, i32 } %hlsl.asuint3, 1
  %13 = insertelement <3 x i32> %8, i32 %11, i64 2
  %14 = insertelement <3 x i32> %9, i32 %12, i64 2
  %add = add <3 x i32> %13, %14
  %conv = uitofp <3 x i32> %add to <3 x float>
  ret <3 x float> %conv
}

attributes #0 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}
!dx.valver = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 8}
!2 = !{!"clang version 20.0.0git (https://github.com/joaosaffran/llvm-project.git 81476c7ad27010600dc4b4be1d66e7c7db7c10fb)"}
