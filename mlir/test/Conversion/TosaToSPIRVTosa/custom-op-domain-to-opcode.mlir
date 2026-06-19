// RUN: mlir-opt --split-input-file --pass-pipeline='builtin.module(tosa-to-spirv-tosa{custom-op-domain-to-opcode=my:custom:0,com.example.accel:42})' %s | FileCheck %s
// RUN: mlir-opt --split-input-file --pass-pipeline='builtin.module(tosa-to-spirv-tosa{custom-op-domain-to-opcode=my:custom:99,my:custom:7,com.example.accel:42})' %s | FileCheck %s --check-prefix=OVERRIDE

//===----------------------------------------------------------------------===//
// Mapped tosa.custom
//===----------------------------------------------------------------------===//

// CHECK: spirv.module @_spirv_tosa_mapped_custom Logical Vulkan
// CHECK: spirv.ARM.Graph @mapped_custom(%arg0: !spirv.arm.tensor<1x16xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}, %arg1: !spirv.arm.tensor<1x16xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) -> (!spirv.arm.tensor<1x16xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}) attributes {entry_point = true} {
func.func @mapped_custom(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>) -> tensor<1x16xf32> {
  // CHECK: %[[OP_NAME:.*]] = spirv.Constant [84 : i8, 101 : i8, 115 : i8, 116 : i8, 79 : i8, 112 : i8] : !spirv.array<6 x i8>
  // CHECK: %[[IMPLEMENTATION_ATTRS:.*]] = spirv.Constant [123 : i8, 34 : i8, 112 : i8, 97 : i8, 114 : i8, 97 : i8, 109 : i8, 34 : i8, 58 : i8, 34 : i8, 118 : i8, 97 : i8, 108 : i8, 117 : i8, 101 : i8, 34 : i8, 125 : i8] : !spirv.array<17 x i8>
  // CHECK: %[[CALL:.*]] = spirv.ExperimentalML.Call opcode = 0, %[[OP_NAME]], %[[IMPLEMENTATION_ATTRS]], %arg0, %arg1 : (!spirv.array<6 x i8>, !spirv.array<17 x i8>, !spirv.arm.tensor<1x16xf32>, !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32>
  // OVERRIDE-LABEL: spirv.ARM.Graph @mapped_custom
  // OVERRIDE: spirv.ExperimentalML.Call opcode = 7,
  %0 = tosa.custom %arg0, %arg1 {domain_name = "my:custom", implementation_attrs = "{\"param\":\"value\"}", operator_name = "TestOp"} : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  // CHECK: spirv.ARM.GraphOutputs %[[CALL]] : !spirv.arm.tensor<1x16xf32>
  return %0 : tensor<1x16xf32>
}

// -----

// CHECK: spirv.module @_spirv_tosa_other_custom Logical Vulkan
// CHECK: spirv.ARM.Graph @other_custom(%arg0: !spirv.arm.tensor<1x16xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<1x16xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
func.func @other_custom(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
  // CHECK: %[[OP_NAME:.*]] = spirv.Constant [69 : i8, 120 : i8, 97 : i8, 109 : i8, 112 : i8, 108 : i8, 101 : i8, 79 : i8, 112 : i8] : !spirv.array<9 x i8>
  // CHECK: %[[IMPLEMENTATION_ATTRS:.*]] = spirv.Constant [123 : i8, 125 : i8] : !spirv.array<2 x i8>
  // CHECK: %[[CALL:.*]] = spirv.ExperimentalML.Call opcode = 42, %[[OP_NAME]], %[[IMPLEMENTATION_ATTRS]], %arg0 : (!spirv.array<9 x i8>, !spirv.array<2 x i8>, !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32>
  %0 = tosa.custom %arg0 {domain_name = "com.example.accel", implementation_attrs = "{}", operator_name = "ExampleOp"} : (tensor<1x16xf32>) -> tensor<1x16xf32>
  // CHECK: spirv.ARM.GraphOutputs %[[CALL]] : !spirv.arm.tensor<1x16xf32>
  return %0 : tensor<1x16xf32>
}

// -----

// CHECK: spirv.module @_spirv_tosa_empty_strings Logical Vulkan
// CHECK: spirv.ARM.Graph @empty_strings(%arg0: !spirv.arm.tensor<1x16xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}) -> (!spirv.arm.tensor<1x16xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}) attributes {entry_point = true} {
func.func @empty_strings(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
  // CHECK: %[[OP_NAME:.*]] = spirv.Constant [0 : i8] : !spirv.array<1 x i8>
  // CHECK: %[[IMPLEMENTATION_ATTRS:.*]] = spirv.Constant [0 : i8] : !spirv.array<1 x i8>
  // CHECK: %[[CALL:.*]] = spirv.ExperimentalML.Call opcode = 0, %[[OP_NAME]], %[[IMPLEMENTATION_ATTRS]], %arg0 : (!spirv.array<1 x i8>, !spirv.array<1 x i8>, !spirv.arm.tensor<1x16xf32>) -> !spirv.arm.tensor<1x16xf32>
  %0 = tosa.custom %arg0 {domain_name = "my:custom", implementation_attrs = "", operator_name = ""} : (tensor<1x16xf32>) -> tensor<1x16xf32>
  // CHECK: spirv.ARM.GraphOutputs %[[CALL]] : !spirv.arm.tensor<1x16xf32>
  return %0 : tensor<1x16xf32>
}
