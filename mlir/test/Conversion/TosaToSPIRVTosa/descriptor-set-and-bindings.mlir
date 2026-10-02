// RUN: mlir-opt --split-input-file --tosa-to-spirv-tosa --verify-diagnostics %s | FileCheck %s

// CHECK-NOT: grapharm.interface_var_abi

// CHECK: spirv.module @_spirv_tosa_default_interface_var_abi Logical Vulkan attributes {spirv.target_env = #spirv.target_env<
// CHECK: spirv.ARM.Graph @default_interface_var_abi(%[[ARG0:.*]]: !spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
func.func @default_interface_var_abi(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  // CHECK: spirv.ARM.GraphOutputs %[[ARG0]] : !spirv.arm.tensor<1xi8>
  return %arg0 : tensor<1xi8>
}

// -----

// CHECK: spirv.module @_spirv_tosa_custom_grapharm_abi Logical Vulkan attributes {spirv.target_env = #spirv.target_env<
// CHECK: spirv.ARM.Graph @custom_grapharm_abi(%[[ARG0:.*]]: !spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(3, 9)>}) -> (!spirv.arm.tensor<1xi8> {spirv.interface_var_abi = #spirv.interface_var_abi<(7, 11)>}) attributes {entry_point = true} {
func.func @custom_grapharm_abi(%arg0: tensor<1xi8> {grapharm.interface_var_abi = #spirv.interface_var_abi<(3, 9)>}) -> (tensor<1xi8> {grapharm.interface_var_abi = #spirv.interface_var_abi<(7, 11)>}) {
  // CHECK: spirv.ARM.GraphOutputs %[[ARG0]] : !spirv.arm.tensor<1xi8>
  return %arg0 : tensor<1xi8>
}
