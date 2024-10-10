; RUN: opt -passes='function(scalarizer<load-store>)' -S -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; CHECK-LABEL: @test_vector_double_split_void
define void @test_vector_double_split_void(<2 x double> noundef %d) {
  ; CHECK: [[ee0:%.*]] = extractelement <2 x double> %d, i64 0
  ; CHECK: [[ie0:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[ee0]])
  ; CHECK: [[ee1:%.*]] = extractelement <2 x double> %d, i64 1
  ; CHECK: [[ie1:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[ee1]])
  ; CHECK-NOT: extractvalue { i32, i32 } {{.*}}, 0
  ; CHECK-NOT: insertelement <2 x i32> {{.*}}, i32 {{.*}}, i64 0
  %hlsl.asuint = call { <2 x i32>, <2 x i32> }  @llvm.dx.splitdouble.v2i32(<2 x double> %d)
  ret void
}

; CHECK-LABEL: @test_vector_double_split
define noundef <3 x i32> @test_vector_double_split(<3 x double> noundef %d) {
  ; CHECK: [[ee0:%.*]] = extractelement <3 x double> %d, i64 0
  ; CHECK: [[ie0:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[ee0]])
  ; CHECK: [[ee1:%.*]] = extractelement <3 x double> %d, i64 1
  ; CHECK: [[ie1:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[ee1]])
  ; CHECK: [[ee2:%.*]] = extractelement <3 x double> %d, i64 2
  ; CHECK: [[ie2:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[ee2]])
  ; CHECK: [[ev00:%.*]] = extractvalue { i32, i32 } [[ie0]], 0
  ; CHECK: [[ev01:%.*]] = extractvalue { i32, i32 } [[ie1]], 0
  ; CHECK: [[ev02:%.*]] = extractvalue { i32, i32 } [[ie2]], 0
  ; CHECK: [[ev10:%.*]] = extractvalue { i32, i32 } [[ie0]], 1
  ; CHECK: [[ev11:%.*]] = extractvalue { i32, i32 } [[ie1]], 1
  ; CHECK: [[ev12:%.*]] = extractvalue { i32, i32 } [[ie2]], 1
  ; CHECK: [[add1:%.*]] = add i32 [[ev00]], [[ev10]]
  ; CHECK: [[add2:%.*]] = add i32 [[ev01]], [[ev11]]
  ; CHECK: [[add3:%.*]] = add i32 [[ev02]], [[ev12]]
  ; CHECK: insertelement <3 x i32> poison, i32 [[add1]], i64 0
  ; CHECK: insertelement <3 x i32> %{{.*}}, i32 [[add2]], i64 1
  ; CHECK: insertelement <3 x i32> %{{.*}}, i32 [[add3]], i64 2
  %hlsl.asuint = call { <3 x i32>, <3 x i32> }  @llvm.dx.splitdouble.v3i32(<3 x double> %d)
  %1 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.asuint, 0
  %2 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.asuint, 1
  %3 = add <3 x i32> %1, %2
  ret <3 x i32> %3
}
