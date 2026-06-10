// RUN: mlir-opt --split-input-file --tosa-to-spirv-tosa --verify-diagnostics %s | FileCheck %s

// CHECK: gpu.module @random_container
gpu.module @random_container {
  // CHECK: spirv.module @_spirv_tosa_nested Logical Vulkan attributes {spirv.target_env = #spirv.target_env<
  // CHECK: spirv.ARM.Graph @nested(%[[ARG0:.*]]: !spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
  func.func @nested(%arg0: tensor<1xi8>) -> tensor<1xi8> {
    // CHECK: spirv.ARM.GraphOutputs %[[ARG0]] : !spirv.arm.tensor<1xi8>
    return %arg0 : tensor<1xi8>
  }
}

// -----

// CHECK: module @random_container {
module @random_container {
  module @yet_anther_random_container {
    // CHECK: gpu.module @another_random_container
    gpu.module @another_random_container {
      // CHECK: spirv.module @_spirv_tosa_nested Logical Vulkan attributes {spirv.target_env = #spirv.target_env<
      // CHECK: spirv.ARM.Graph @nested(%[[ARG0:.*]]: !spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
      func.func @nested(%arg0: tensor<1xi8>) -> tensor<1xi8> {
        // CHECK: spirv.ARM.GraphOutputs %[[ARG0]] : !spirv.arm.tensor<1xi8>
        return %arg0 : tensor<1xi8>
      }
    }
  }
}
