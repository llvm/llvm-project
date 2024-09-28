; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; RUN: opt -S --scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure DXILOpLowering is correctly generating the dxil op code call, with and without scalarizer.

; CHECK-LABEL: define noundef float @test_scalar_double_split
define noundef float @test_scalar_double_split(double noundef %D) local_unnamed_addr {
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

declare <2 x i32> @llvm.dx.splitdouble.v2i32(double) #1


; CHECK-LABEL: define noundef <3 x float> @test_vector_double_split
define noundef <3 x float> @test_vector_double_split(<3 x double> noundef %D) local_unnamed_addr {
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
