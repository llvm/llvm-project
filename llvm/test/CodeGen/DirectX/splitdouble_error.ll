; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation splitdouble doesn't support vector types.
; XFAIL: * 

define noundef <3 x i32> @test_vector_double_split(<3 x double> noundef %D) local_unnamed_addr {
entry:
  %hlsl.splitdouble = call { <3 x i32>, <3 x i32> } @llvm.dx.splitdouble.v3i32(<3 x double> %D)
  %0 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.splitdouble, 0
  %1 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.splitdouble, 1
  %add = add <3 x i32> %0, %1
  ret <3 x i32> %add
}

declare { <3 x i32>, <3 x i32> } @llvm.dx.splitdouble.v3i32(<3 x double>)
