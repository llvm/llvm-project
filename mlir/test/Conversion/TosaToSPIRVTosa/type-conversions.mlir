// RUN: mlir-opt --split-input-file --tosa-to-spirv-tosa --verify-diagnostics %s | FileCheck %s

// CHECK: spirv.module @_spirv_tosa_i48_to_i64 Logical Vulkan attributes {spirv.target_env = #spirv.target_env<
// CHECK: spirv.ARM.Graph @i48_to_i64(%[[ARG0:.*]]: !spirv.arm.tensor<1x2x3x4xi64> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<1x2x3x4xi64> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
func.func @i48_to_i64(%arg0: tensor<1x2x3x4xi48>) -> tensor<1x2x3x4xi48> {
  // CHECK: spirv.ARM.GraphOutputs %[[ARG0]] : !spirv.arm.tensor<1x2x3x4xi64>
  return %arg0 : tensor<1x2x3x4xi48>
}

// -----

// CHECK: spirv.module @_spirv_tosa_i4_to_i8 Logical Vulkan attributes {spirv.target_env = #spirv.target_env<
// CHECK: spirv.ARM.Graph @i4_to_i8(%[[ARG0:.*]]: !spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
func.func @i4_to_i8(%arg0: tensor<1xi4>) -> tensor<1xi4> {
  // CHECK: spirv.ARM.GraphOutputs %[[ARG0]] : !spirv.arm.tensor<1xi8>
  return %arg0 : tensor<1xi4>
}

// -----

// CHECK: spirv.module @_spirv_tosa_scalar_tensor_to_rank1_tensor Logical Vulkan attributes {spirv.target_env = #spirv.target_env<
// CHECK: spirv.ARM.Graph @scalar_tensor_to_rank1_tensor(%[[ARG0:.*]]: !spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
func.func @scalar_tensor_to_rank1_tensor(%arg0: tensor<i8>) -> tensor<i8> {
  // CHECK: spirv.ARM.GraphOutputs %[[ARG0]] : !spirv.arm.tensor<1xi8>
  return %arg0 : tensor<i8>
}

// -----

// expected-error@below {{failed to convert function argument types}}
func.func @zero_sized_tensor(%arg0: tensor<0xi8>) -> tensor<0xi8> {
  return %arg0 : tensor<0xi8>
}

// -----

// expected-error@below {{failed to convert function argument types}}
func.func @mixed_zero_sized_tensor(%arg0: tensor<1x0x2xi8>) -> tensor<1x0x2xi8> {
  return %arg0 : tensor<1x0x2xi8>
}

// -----

// CHECK: spirv.module @_spirv_tosa_partially_shaped_tensor_to_unshaped_tensor Logical Vulkan attributes {spirv.target_env = #spirv.target_env<
// CHECK: spirv.ARM.Graph @partially_shaped_tensor_to_unshaped_tensor(%[[ARG0:.*]]: !spirv.arm.tensor<?x?xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<?x?xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
func.func @partially_shaped_tensor_to_unshaped_tensor(%arg0: tensor<1x?xi8>) -> tensor<1x?xi8> {
  // CHECK: spirv.ARM.GraphOutputs %[[ARG0]] : !spirv.arm.tensor<?x?xi8>
  return %arg0 : tensor<1x?xi8>
}

// -----

// CHECK: spirv.module @_spirv_tosa_unranked_tensor Logical Vulkan attributes {spirv.target_env = #spirv.target_env<
// CHECK: spirv.ARM.Graph @unranked_tensor(%[[ARG0:.*]]: !spirv.arm.tensor<*xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<*xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
func.func @unranked_tensor(%arg0: tensor<*xi8>) -> tensor<*xi8> {
  // CHECK: spirv.ARM.GraphOutputs %[[ARG0]] : !spirv.arm.tensor<*xi8>
  return %arg0 : tensor<*xi8>
}

// -----

// expected-error@below {{'builtin.module' op requires GraphARM capability and SPV_ARM_graph/SPV_ARM_tensors extensions in spirv.target_env}}
module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>} {
  func.func @unsupported_target_env(%arg0: tensor<1xi8>) -> tensor<1xi8> {
    return %arg0 : tensor<1xi8>
  }
}
