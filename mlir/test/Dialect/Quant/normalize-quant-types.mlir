// RUN: mlir-opt %s --normalize-quant-types --split-input-file | FileCheck %s

// CHECK-LABEL: @callee(
// CHECK-SAME: [[PER_TENSOR:tensor<\?x\?x!quant.uniform<i8:f32, 2.000000e\+00:127>>]],
// CHECK-SAME: [[PER_TENSOR]]
// CHECK-SAME: ([[PER_TENSOR]], [[PER_TENSOR]])
// CHECK-LABEL: @normalize_quant_types_to_per_tensor
// CHECK-SAME: %[[ARG_0:.*]]: [[PER_TENSOR:tensor<\?x\?x!quant.uniform<i8:f32, 2.000000e\+00:127>>]],
// CHECK-SAME: %[[ARG_1:.*]]: [[PER_TENSOR]]
// CHECK-SAME: ([[PER_TENSOR]], [[PER_TENSOR]])
// CHECK: %[[TEMP_0:.*]] = "test.custom_op"(%[[ARG_0]]) : ([[PER_TENSOR]]) -> [[PER_TENSOR]]
// CHECK: %[[TEMP_1:.*]] = "test.custom_op"(%[[ARG_1]]) : ([[PER_TENSOR]]) -> [[PER_TENSOR]]
// CHECK: %[[TEMP_3:.*]]:2 = call @callee(%[[TEMP_0]], %[[TEMP_1]])
// CHECK: return %[[TEMP_3]]#0, %[[TEMP_3]]#1 : [[PER_TENSOR]], [[PER_TENSOR]]

!qalias1 = !quant.uniform<i8:f32:{}, {{2.0:127}}>
!qalias2 = !quant.uniform<i8:f32:{0:1,1:4}, {{2.0:127}}>

func.func private @callee(tensor<?x?x!qalias1>, tensor<?x?x!qalias2>) -> (tensor<?x?x!qalias1>, tensor<?x?x!qalias2>)

func.func @normalize_quant_types_to_per_tensor(%arg0: tensor<?x?x!qalias1>,
    %arg1: tensor<?x?x!qalias2>) -> (tensor<?x?x!qalias1>, tensor<?x?x!qalias2>) {
  %0 = "test.custom_op"(%arg0) : (tensor<?x?x!qalias1>) -> tensor<?x?x!qalias1>
  %1 = "test.custom_op"(%arg1) : (tensor<?x?x!qalias2>) -> tensor<?x?x!qalias2>
  %3:2 = func.call @callee(%0, %1) : (tensor<?x?x!qalias1>, tensor<?x?x!qalias2>) -> (tensor<?x?x!qalias1>, tensor<?x?x!qalias2>)
  return %3#0, %3#1 : tensor<?x?x!qalias1>, tensor<?x?x!qalias2>
}

// -----

// CHECK-LABEL: @normalize_quant_types_to_per_axis
// CHECK-SAME: %[[ARG_0:.*]]: [[PER_AXIS:tensor<\?x\?x!quant.uniform<i8:f32:0, \{2.000000e\+00:127,3.000000e\+00:127\}>>]],
// CHECK-SAME: %[[ARG_1:.*]]: [[PER_AXIS]]
// CHECK-SAME: ([[PER_AXIS]], [[PER_AXIS]])
// CHECK: %[[TEMP_0:.*]] = "test.custom_op"(%[[ARG_0]]) : ([[PER_AXIS]]) -> [[PER_AXIS]]
// CHECK: %[[TEMP_1:.*]] = "test.custom_op"(%[[ARG_1]]) : ([[PER_AXIS]]) -> [[PER_AXIS]]
// CHECK: %[[TEMP_3:.*]]:2 = call @callee(%[[TEMP_0]], %[[TEMP_1]])
// CHECK: return %[[TEMP_3]]#0, %[[TEMP_3]]#1 : [[PER_AXIS]], [[PER_AXIS]]

!qalias1 = !quant.uniform<i8:f32:{0:1}, {{2.0:127}, {3.0:127}}>
!qalias2 = !quant.uniform<i8:f32:{0:1,1:4}, {{2.0:127}, {3.0:127}}>

func.func private @callee(tensor<?x?x!qalias1>, tensor<?x?x!qalias2>) -> (tensor<?x?x!qalias1>, tensor<?x?x!qalias2>)

func.func @normalize_quant_types_to_per_axis(%arg0: tensor<?x?x!qalias1>,
    %arg1: tensor<?x?x!qalias2>) -> (tensor<?x?x!qalias1>, tensor<?x?x!qalias2>) {
  %0 = "test.custom_op"(%arg0) : (tensor<?x?x!qalias1>) -> tensor<?x?x!qalias1>
  %1 = "test.custom_op"(%arg1) : (tensor<?x?x!qalias2>) -> tensor<?x?x!qalias2>
  %3:2 = func.call @callee(%0, %1) : (tensor<?x?x!qalias1>, tensor<?x?x!qalias2>) -> (tensor<?x?x!qalias1>, tensor<?x?x!qalias2>)
  return %3#0, %3#1 : tensor<?x?x!qalias1>, tensor<?x?x!qalias2>
}
