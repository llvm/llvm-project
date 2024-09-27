; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; RUN: opt -S --scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure DXILOpLowering is correctly generating the dxil op code call, with and without scalarizer.

; CHECK-LABEL: define noundef float @test_scalar_double_split
define noundef float @test_scalar_double_split(double noundef %D) local_unnamed_addr {
entry:
  ; CHECK: [[CALL:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double %D)
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  %hlsl.asuint = call <2 x i32> @llvm.dx.splitdouble.v2i32(double %D)
  %1 = extractelement <2 x i32> %hlsl.asuint, i64 0
  %2 = extractelement <2 x i32> %hlsl.asuint, i64 1
  %add = add i32 %1, %2
  %conv = uitofp i32 %add to float
  ret float %conv
}

declare <2 x i32> @llvm.dx.splitdouble.v2i32(double) #1


; CHECK-LABEL: define noundef <3 x float> @test_vector_double_split
define noundef <3 x float> @test_vector_double_split(<3 x double> noundef %D) local_unnamed_addr {
entry:
  %0 = extractelement <3 x double> %D, i64 0
  ; CHECK-COUNT-3: [[CALL:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double {{.*}})
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  %hlsl.asuint = call <2 x i32> @llvm.dx.splitdouble.v2i32(double %0)
  %1 = extractelement <2 x i32> %hlsl.asuint, i64 0
  %2 = extractelement <2 x i32> %hlsl.asuint, i64 1
  %3 = insertelement <3 x i32> poison, i32 %1, i64 0
  %4 = insertelement <3 x i32> poison, i32 %2, i64 0
  %5 = extractelement <3 x double> %D, i64 1
  %hlsl.asuint2 = call <2 x i32> @llvm.dx.splitdouble.v2i32(double %5)
  %6 = extractelement <2 x i32> %hlsl.asuint2, i64 0
  %7 = extractelement <2 x i32> %hlsl.asuint2, i64 1
  %8 = insertelement <3 x i32> %3, i32 %6, i64 1
  %9 = insertelement <3 x i32> %4, i32 %7, i64 1
  %10 = extractelement <3 x double> %D, i64 2
  %hlsl.asuint3 = call <2 x i32> @llvm.dx.splitdouble.v2i32(double %10)
  %11 = extractelement <2 x i32> %hlsl.asuint3, i64 0
  %12 = extractelement <2 x i32> %hlsl.asuint3, i64 1
  %13 = insertelement <3 x i32> %8, i32 %11, i64 2
  %14 = insertelement <3 x i32> %9, i32 %12, i64 2
  %add = add <3 x i32> %13, %14
  %conv = uitofp <3 x i32> %add to <3 x float>
  ret <3 x float> %conv
}
