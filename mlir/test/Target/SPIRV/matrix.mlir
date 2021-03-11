// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @matrix_access_chain
  spv.func @matrix_access_chain(%arg0 : !spv.ptr<!spv.matrix<3 x vector<3xf32>>, Function>, %arg1 : i32) -> !spv.ptr<vector<3xf32>, Function> "None" {
    // CHECK: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.matrix<3 x vector<3xf32>>, Function>
    %0 = spv.AccessChain %arg0[%arg1] : !spv.ptr<!spv.matrix<3 x vector<3xf32>>,Function>, i32
    spv.ReturnValue %0 : !spv.ptr<vector<3xf32>, Function>
  }

  // CHECK-LABEL: @matrix_times_scalar_1
  spv.func @matrix_times_scalar_1(%arg0 : !spv.matrix<3 x vector<3xf32>>, %arg1 : f32) -> !spv.matrix<3 x vector<3xf32>> "None" {
    // CHECK: {{%.*}} = spv.MatrixTimesScalar {{%.*}}, {{%.*}} : !spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<3 x vector<3xf32>>
    %result = spv.MatrixTimesScalar %arg0, %arg1 : !spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<3 x vector<3xf32>>
    spv.ReturnValue %result : !spv.matrix<3 x vector<3xf32>>
  }

  // CHECK-LABEL: @matrix_times_scalar_2
  spv.func @matrix_times_scalar_2(%arg0 : !spv.matrix<3 x vector<3xf16>>, %arg1 : f16) -> !spv.matrix<3 x vector<3xf16>> "None" {
    // CHECK: {{%.*}} = spv.MatrixTimesScalar {{%.*}}, {{%.*}} : !spv.matrix<3 x vector<3xf16>>, f16 -> !spv.matrix<3 x vector<3xf16>>
    %result = spv.MatrixTimesScalar %arg0, %arg1 : !spv.matrix<3 x vector<3xf16>>, f16 -> !spv.matrix<3 x vector<3xf16>>
    spv.ReturnValue %result : !spv.matrix<3 x vector<3xf16>>

  }

  // CHECK-LABEL: @matrix_transpose_1
  spv.func @matrix_transpose_1(%arg0 : !spv.matrix<3 x vector<2xf32>>) -> !spv.matrix<2 x vector<3xf32>> "None" {
    // CHECK: {{%.*}} = spv.Transpose {{%.*}} : !spv.matrix<3 x vector<2xf32>> -> !spv.matrix<2 x vector<3xf32>>
    %result = spv.Transpose %arg0 : !spv.matrix<3 x vector<2xf32>> -> !spv.matrix<2 x vector<3xf32>>
    spv.ReturnValue %result : !spv.matrix<2 x vector<3xf32>>
  }

  // CHECK-LABEL: @matrix_times_matrix_1
  spv.func @matrix_times_matrix_1(%arg0: !spv.matrix<3 x vector<3xf32>>, %arg1: !spv.matrix<3 x vector<3xf32>>) -> !spv.matrix<3 x vector<3xf32>> "None"{
    // CHECK: {{%.*}} = spv.MatrixTimesMatrix {{%.*}}, {{%.*}} : !spv.matrix<3 x vector<3xf32>>, !spv.matrix<3 x vector<3xf32>> -> !spv.matrix<3 x vector<3xf32>>
    %result = spv.MatrixTimesMatrix %arg0, %arg1 : !spv.matrix<3 x vector<3xf32>>, !spv.matrix<3 x vector<3xf32>> -> !spv.matrix<3 x vector<3xf32>>
    spv.ReturnValue %result : !spv.matrix<3 x vector<3xf32>>
  }

  // CHECK-LABEL: @matrix_times_matrix_2
  spv.func @matrix_times_matrix_2(%arg0: !spv.matrix<3 x vector<2xf32>>, %arg1: !spv.matrix<2 x vector<3xf32>>) -> !spv.matrix<2 x vector<2xf32>> "None"{
    // CHECK: {{%.*}} = spv.MatrixTimesMatrix {{%.*}}, {{%.*}} : !spv.matrix<3 x vector<2xf32>>, !spv.matrix<2 x vector<3xf32>> -> !spv.matrix<2 x vector<2xf32>>
    %result = spv.MatrixTimesMatrix %arg0, %arg1 : !spv.matrix<3 x vector<2xf32>>, !spv.matrix<2 x vector<3xf32>> -> !spv.matrix<2 x vector<2xf32>>
    spv.ReturnValue %result : !spv.matrix<2 x vector<2xf32>>
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: spv.GlobalVariable {{@.*}} : !spv.ptr<!spv.matrix<3 x vector<3xf32>>, StorageBuffer>
  spv.GlobalVariable @var0 : !spv.ptr<!spv.matrix<3 x vector<3xf32>>, StorageBuffer>

  // CHECK: spv.GlobalVariable {{@.*}} : !spv.ptr<!spv.matrix<2 x vector<3xf32>>, StorageBuffer>
  spv.GlobalVariable @var1 : !spv.ptr<!spv.matrix<2 x vector<3xf32>>, StorageBuffer>

  // CHECK: spv.GlobalVariable {{@.*}} : !spv.ptr<!spv.matrix<4 x vector<4xf16>>, StorageBuffer>
  spv.GlobalVariable @var2 : !spv.ptr<!spv.matrix<4 x vector<4xf16>>, StorageBuffer>
}
