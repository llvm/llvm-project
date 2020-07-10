// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @matrix_times_scalar
  spv.func @matrix_times_scalar(%arg0 : !spv.matrix<3 x vector<3xf32>>, %arg1 : f32) -> !spv.matrix<3 x vector<3xf32>> "None" {
    // CHECK: {{%.*}} = spv.MatrixTimesScalar {{%.*}}, {{%.*}} : !spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<3 x vector<3xf32>>
    %result = spv.MatrixTimesScalar %arg0, %arg1 : !spv.matrix<3 x vector<3xf32>>, f32 -> !spv.matrix<3 x vector<3xf32>>
    spv.ReturnValue %result : !spv.matrix<3 x vector<3xf32>>
  }

  // CHECK-LABEL: @matrix_transpose_1
  spv.func @matrix_transpose_1(%arg0 : !spv.matrix<3 x vector<2xf32>>) -> !spv.matrix<2 x vector<3xf32>> "None" {
    // CHECK: {{%.*}} = spv.Transpose {{%.*}} : !spv.matrix<3 x vector<2xf32>> -> !spv.matrix<2 x vector<3xf32>>
    %result = spv.Transpose %arg0 : !spv.matrix<3 x vector<2xf32>> -> !spv.matrix<2 x vector<3xf32>>
    spv.ReturnValue %result : !spv.matrix<2 x vector<3xf32>>
  }

  // CHECK-LABEL: @matrix_transpose_2
  spv.func @matrix_transpose_2(%arg0 : !spv.matrix<3 x vector<3xf32>>) -> !spv.matrix<3 x vector<3xf32>> "None" {
    // CHECK: {{%.*}} = spv.Transpose {{%.*}} : !spv.matrix<3 x vector<3xf32>> -> !spv.matrix<3 x vector<3xf32>>
    %result = spv.Transpose %arg0 : !spv.matrix<3 x vector<3xf32>> -> !spv.matrix<3 x vector<3xf32>>
    spv.ReturnValue %result : !spv.matrix<3 x vector<3xf32>>
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

// -----

func @transpose_op_shape_mismatch_1(%arg0 : !spv.matrix<3 x vector<4xf32>>) -> () {
   // expected-error @+1 {{input matrix rows count must be equal to output matrix columns count}}
   %result = spv.Transpose %arg0 : !spv.matrix<3 x vector<4xf32>> -> !spv.matrix<3 x vector<3xf32>>
   spv.Return
}

// -----

func @transpose_op_shape_mismatch_2(%arg0 : !spv.matrix<3 x vector<4xf32>>) -> () {
   // expected-error @+1 {{input matrix rows count must be equal to output matrix columns count}}
   %result = spv.Transpose %arg0 : !spv.matrix<3 x vector<4xf32>> -> !spv.matrix<2 x vector<4xf32>>
   spv.Return
}

// -----

func @transpose_op_type_mismatch(%arg0 : !spv.matrix<3 x vector<4xf32>>) -> () {
   // expected-error @+1 {{input and output matrices must have the same component type}}
   %result = spv.Transpose %arg0 : !spv.matrix<3 x vector<4xf32>> -> !spv.matrix<4 x vector<3xf16>>
   spv.Return
}

// -----

func @matrix_times_matrix_invalid_input_shape_1(%arg0 : !spv.matrix<3 x vector<2xf32>>, %arg1 : !spv.matrix<2 x vector<3xf32>>){
   // expected-error @+1 {{right and result matrices must have equal columns' count}}
   %result = spv.MatrixTimesMatrix %arg0, %arg1 : !spv.matrix<3 x vector<2xf32>>, !spv.matrix<2 x vector<3xf32>> -> !spv.matrix<3 x vector<2xf32>>
}

// -----

func @matrix_times_matrix_invalid_input_shape_2(%arg0 : !spv.matrix<3 x vector<2xf32>>, %arg1 : !spv.matrix<2 x vector<3xf32>>){
   // expected-error @+1 {{left and result matrices must have equal rows' count}}
   %result = spv.MatrixTimesMatrix %arg0, %arg1 : !spv.matrix<3 x vector<2xf32>>, !spv.matrix<2 x vector<3xf32>> -> !spv.matrix<2 x vector<3xf32>>
}

// -----

func @matrix_times_matrix_inputs_shape_mismatch(%arg0 : !spv.matrix<3 x vector<2xf32>>, %arg1 : !spv.matrix<2 x vector<2xf32>>){
   // expected-error @+1 {{left matrix columns' count must be equal to the right matrix rows' count}}
   %result = spv.MatrixTimesMatrix %arg0, %arg1 : !spv.matrix<3 x vector<2xf32>>, !spv.matrix<2 x vector<2xf32>> -> !spv.matrix<2 x vector<2xf32>>
}

// -----

func @matrix_times_matrix_component_type_mismatch_1(%arg0 : !spv.matrix<3 x vector<3xf32>>, %arg1 : !spv.matrix<3x vector<3xf32>>){
   // expected-error @+1 {{right and result matrices' component type must be the same}}
   %result = spv.MatrixTimesMatrix %arg0, %arg1 : !spv.matrix<3 x vector<3xf32>>, !spv.matrix<3 x vector<3xf32>> -> !spv.matrix<3 x vector<3xf64>>
}


// -----

func @matrix_times_matrix_component_type_mismatch_2(%arg0 : !spv.matrix<3 x vector<3xf64>>, %arg1 : !spv.matrix<3x vector<3xf32>>){
   // expected-error @+1 {{left and result matrices' component type must be the same}}
   %result = spv.MatrixTimesMatrix %arg0, %arg1 : !spv.matrix<3 x vector<3xf64>>, !spv.matrix<3 x vector<3xf32>> -> !spv.matrix<3 x vector<3xf32>>
}
