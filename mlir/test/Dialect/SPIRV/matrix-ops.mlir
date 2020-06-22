// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @matrix_times_scalar
  spv.func @matrix_times_scalar_1(%arg0 : !spv.matrix<3 x vector<3xf32>>, %arg1 : f32) -> !spv.matrix<3 x vector<3xf32>> "None" {
    // CHECK: {{%.*}} = spv.MatrixTimesScalar {{%.*}}, {{%.*}} : !spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<3 x vector<3xf32>>
    %result = spv.MatrixTimesScalar %arg0, %arg1 : !spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<3 x vector<3xf32>>
    spv.ReturnValue %result : !spv.matrix<3 x vector<3xf32>>
  }
}

// -----

func @input_type_mismatch(%arg0 : !spv.matrix<3 x vector<3xf32>>, %arg1 : f16) -> () {
  // expected-error @+1 {{input matrix components' type and scaling value must have the same type}}
  %result = spv.MatrixTimesScalar %arg0, %arg1 : !spv.matrix<3 x vector<3xf32>>, f16 -> !spv.matrix<3 x vector<3xf32>>
}

// -----

func @input_type_mismatch(%arg0 : !spv.matrix<3 x vector<3xf32>>, %arg1 : f64) -> () {
  // expected-error @+1 {{input matrix components' type and scaling value must have the same type}}
  %result = spv.MatrixTimesScalar %arg0, %arg1 : !spv.matrix<3 x vector<3xf32>>, f64 -> !spv.matrix<3 x vector<3xf32>>
}

// -----

func @input_output_component_type_mismatch(%arg0 : !spv.matrix<3 x vector<3xf32>>, %arg1 : f32) -> () {
   // expected-error @+1 {{input and result matrices' columns must have the same component type}}
   %result = spv.MatrixTimesScalar %arg0, %arg1 : !spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<3 x vector<3xf64>>
}

// -----

func @input_output_size_mismatch(%arg0 : !spv.matrix<3 x vector<3xf32>>, %arg1 : f32) -> () {
   // expected-error @+1 {{input and result matrices must have the same number of columns}}
   %result = spv.MatrixTimesScalar %arg0, %arg1 : !spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<4 x vector<3xf32>>
}



