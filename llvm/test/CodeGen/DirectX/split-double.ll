
; RUN: opt -S -scalarizer -scalarize-load-store -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define noundef <3 x i32> @test_vector_double_split(<3 x double> noundef %d) local_unnamed_addr {
    %hlsl.asuint = call { <3 x i32>, <3 x i32> }  @llvm.dx.splitdouble.v3i32(<3 x double> %d)
    %1 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.asuint, 0
    %2 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.asuint, 1
    %3 = add <3 x i32> %1, %2
    ret <3 x i32> %3
}
