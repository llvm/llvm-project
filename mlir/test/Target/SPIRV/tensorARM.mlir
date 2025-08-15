// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, TensorsARM], [SPV_ARM_tensors]> {
  // CHECK: spirv.func @shaped_int_arm_tensor(%arg0: !spirv.arm.tensor<2xi32>) "None" {
  spirv.func @shaped_int_arm_tensor(%arg0 : !spirv.arm.tensor<2xi32>) "None" {
    spirv.Return
  }

// -----

  // CHECK: spirv.func @shaped_rank2_int_arm_tensor(%arg0: !spirv.arm.tensor<2x3xi32>) "None" {
  spirv.func @shaped_rank2_int_arm_tensor(%arg0 : !spirv.arm.tensor<2x3xi32>) "None" {
    spirv.Return
  }

// -----

  // CHECK: spirv.func @ui64_arm_tensor_const() -> !spirv.arm.tensor<3xi64> "None" {
  spirv.func @ui64_arm_tensor_const() -> !spirv.arm.tensor<3xui64> "None" {
    // CHECK: spirv.Constant dense<[5, 6, 7]> : !spirv.arm.tensor<3xi64>
    %0 = spirv.Constant dense<[5, 6, 7]> : !spirv.arm.tensor<3xui64>

    spirv.ReturnValue %0: !spirv.arm.tensor<3xui64>
  }

// -----

  // CHECK: spirv.func @si32_arm_tensor_const() -> !spirv.arm.tensor<3xsi32> "None" {
  spirv.func @si32_arm_tensor_const() -> !spirv.arm.tensor<3xsi32> "None" {
    // CHECK: spirv.Constant dense<[5, 6, 7]> : !spirv.arm.tensor<3xsi32>
    %0 = spirv.Constant dense<[5, 6, 7]> : !spirv.arm.tensor<3xsi32>

    spirv.ReturnValue %0 : !spirv.arm.tensor<3xsi32>
  }

// -----

  // CHECK: spirv.func @float_arm_tensor_const() -> !spirv.arm.tensor<3xf32> "None" {
  spirv.func @float_arm_tensor_const() -> !spirv.arm.tensor<3xf32> "None" {
    // CHECK: spirv.Constant dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : !spirv.arm.tensor<3xf32>
    %0 = spirv.Constant dense<[3., 4., 5.]> : !spirv.arm.tensor<3xf32>

    spirv.ReturnValue %0 : !spirv.arm.tensor<3xf32>
  }

// -----

  // CHECK: spirv.func @unranked_int_arm_tensor(%arg0: !spirv.arm.tensor<*xi32>) "None" {
  spirv.func @unranked_int_arm_tensor(%arg0 : !spirv.arm.tensor<*xi32>) "None" {
    spirv.Return
  }

// -----

  // CHECK: spirv.func @unshaped_int_arm_tensor(%arg0: !spirv.arm.tensor<?xi32>) "None" {
  spirv.func @unshaped_int_arm_tensor(%arg0 : !spirv.arm.tensor<?xi32>) "None" {
    spirv.Return
  }

// -----

  // CHECK: spirv.func @unshaped_int_arm_tensor_2(%arg0: !spirv.arm.tensor<?x?xi32>) "None" {
  spirv.func @unshaped_int_arm_tensor_2(%arg0 : !spirv.arm.tensor<?x?xi32>) "None" {
    spirv.Return
  }
}
