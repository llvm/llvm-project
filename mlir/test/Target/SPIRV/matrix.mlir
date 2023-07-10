// RUN: mlir-translate -no-implicit-module -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @matrix_access_chain
  spirv.func @matrix_access_chain(%arg0 : !spirv.ptr<!spirv.matrix<3 x vector<3xf32>>, Function>, %arg1 : i32) -> !spirv.ptr<vector<3xf32>, Function> "None" {
    // CHECK: {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}] : !spirv.ptr<!spirv.matrix<3 x vector<3xf32>>, Function>
    %0 = spirv.AccessChain %arg0[%arg1] : !spirv.ptr<!spirv.matrix<3 x vector<3xf32>>,Function>, i32
    spirv.ReturnValue %0 : !spirv.ptr<vector<3xf32>, Function>
  }

  // CHECK-LABEL: @matrix_times_scalar_1
  spirv.func @matrix_times_scalar_1(%arg0 : !spirv.matrix<3 x vector<3xf32>>, %arg1 : f32) -> !spirv.matrix<3 x vector<3xf32>> "None" {
    // CHECK: {{%.*}} = spirv.MatrixTimesScalar {{%.*}}, {{%.*}} : !spirv.matrix<3 x vector<3xf32>>, f32
    %result = spirv.MatrixTimesScalar %arg0, %arg1 : !spirv.matrix<3 x vector<3xf32>>, f32
    spirv.ReturnValue %result : !spirv.matrix<3 x vector<3xf32>>
  }

  // CHECK-LABEL: @matrix_times_scalar_2
  spirv.func @matrix_times_scalar_2(%arg0 : !spirv.matrix<3 x vector<3xf16>>, %arg1 : f16) -> !spirv.matrix<3 x vector<3xf16>> "None" {
    // CHECK: {{%.*}} = spirv.MatrixTimesScalar {{%.*}}, {{%.*}} : !spirv.matrix<3 x vector<3xf16>>, f16
    %result = spirv.MatrixTimesScalar %arg0, %arg1 : !spirv.matrix<3 x vector<3xf16>>, f16
    spirv.ReturnValue %result : !spirv.matrix<3 x vector<3xf16>>
  }

  // CHECK-LABEL: @matrix_times_scalar_3
  spirv.func @matrix_times_scalar_3(%arg0 : !spirv.NV.coopmatrix<16x16xf16, Subgroup>, %arg1 : f16) -> !spirv.NV.coopmatrix<16x16xf16, Subgroup> "None" {
    // CHECK: {{%.*}} = spirv.MatrixTimesScalar {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<16x16xf16, Subgroup>, f16
    %result = spirv.MatrixTimesScalar %arg0, %arg1 : !spirv.NV.coopmatrix<16x16xf16, Subgroup>, f16
    spirv.ReturnValue %result : !spirv.NV.coopmatrix<16x16xf16, Subgroup>
  }

  // CHECK-LABEL: @matrix_transpose_1
  spirv.func @matrix_transpose_1(%arg0 : !spirv.matrix<3 x vector<2xf32>>) -> !spirv.matrix<2 x vector<3xf32>> "None" {
    // CHECK: {{%.*}} = spirv.Transpose {{%.*}} : !spirv.matrix<3 x vector<2xf32>> -> !spirv.matrix<2 x vector<3xf32>>
    %result = spirv.Transpose %arg0 : !spirv.matrix<3 x vector<2xf32>> -> !spirv.matrix<2 x vector<3xf32>>
    spirv.ReturnValue %result : !spirv.matrix<2 x vector<3xf32>>
  }

  // CHECK-LABEL: @matrix_times_matrix_1
  spirv.func @matrix_times_matrix_1(%arg0: !spirv.matrix<3 x vector<3xf32>>, %arg1: !spirv.matrix<3 x vector<3xf32>>) -> !spirv.matrix<3 x vector<3xf32>> "None"{
    // CHECK: {{%.*}} = spirv.MatrixTimesMatrix {{%.*}}, {{%.*}} : !spirv.matrix<3 x vector<3xf32>>, !spirv.matrix<3 x vector<3xf32>> -> !spirv.matrix<3 x vector<3xf32>>
    %result = spirv.MatrixTimesMatrix %arg0, %arg1 : !spirv.matrix<3 x vector<3xf32>>, !spirv.matrix<3 x vector<3xf32>> -> !spirv.matrix<3 x vector<3xf32>>
    spirv.ReturnValue %result : !spirv.matrix<3 x vector<3xf32>>
  }

  // CHECK-LABEL: @matrix_times_matrix_2
  spirv.func @matrix_times_matrix_2(%arg0: !spirv.matrix<3 x vector<2xf32>>, %arg1: !spirv.matrix<2 x vector<3xf32>>) -> !spirv.matrix<2 x vector<2xf32>> "None"{
    // CHECK: {{%.*}} = spirv.MatrixTimesMatrix {{%.*}}, {{%.*}} : !spirv.matrix<3 x vector<2xf32>>, !spirv.matrix<2 x vector<3xf32>> -> !spirv.matrix<2 x vector<2xf32>>
    %result = spirv.MatrixTimesMatrix %arg0, %arg1 : !spirv.matrix<3 x vector<2xf32>>, !spirv.matrix<2 x vector<3xf32>> -> !spirv.matrix<2 x vector<2xf32>>
    spirv.ReturnValue %result : !spirv.matrix<2 x vector<2xf32>>
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: spirv.GlobalVariable {{@.*}} : !spirv.ptr<!spirv.matrix<3 x vector<3xf32>>, StorageBuffer>
  spirv.GlobalVariable @var0 : !spirv.ptr<!spirv.matrix<3 x vector<3xf32>>, StorageBuffer>

  // CHECK: spirv.GlobalVariable {{@.*}} : !spirv.ptr<!spirv.matrix<2 x vector<3xf32>>, StorageBuffer>
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.matrix<2 x vector<3xf32>>, StorageBuffer>

  // CHECK: spirv.GlobalVariable {{@.*}} : !spirv.ptr<!spirv.matrix<4 x vector<4xf16>>, StorageBuffer>
  spirv.GlobalVariable @var2 : !spirv.ptr<!spirv.matrix<4 x vector<4xf16>>, StorageBuffer>
}
