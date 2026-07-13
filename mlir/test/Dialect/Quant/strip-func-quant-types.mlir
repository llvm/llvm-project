// RUN: mlir-opt %s --strip-func-quant-types --split-input-file | FileCheck %s

// CHECK-LABEL: @strip_operands
// CHECK-SAME: %[[ARG_0:.*]]: i8
// CHECK-SAME: %[[ARG_1:.*]]: i16
// CHECK-SAME: %[[ARG_2:.*]]: f32

// CHECK: %[[ARG_0_CAST:.*]] = quant.scast %[[ARG_1]] : i16 to !quant.uniform<{{.*}}>
// CHECK: %[[ARG_1_CAST:.*]] = quant.scast %[[ARG_0]] : i8 to !quant.uniform<{{.*}}>

// CHECK: "test.custom_op"(%[[ARG_1_CAST]])
// CHECK: "test.custom_op"(%[[ARG_0_CAST]])
// CHECK: "test.custom_op"(%[[ARG_2]])

!qalias = !quant.uniform<i8:f32, 2.0:128>
!qalias1 = !quant.uniform<i16:f32, 3.0:128>

func.func @strip_operands(%arg0: !qalias, %arg1: !qalias1, %arg2: f32) {
  "test.custom_op"(%arg0) : (!qalias) -> tensor<4x!qalias>
  "test.custom_op"(%arg1) : (!qalias1) -> tensor<?x!qalias1>
  "test.custom_op"(%arg2) : (f32) -> tensor<4xf32>
}

// -----

// CHECK-LABEL: @strip_results
// CHECK-SAME: tensor<4xi8>, tensor<?xi16>, tensor<*xi8>, tensor<4xf32>

// CHECK: %[[RESULT_0:.*]] = "test.custom_op"()
// CHECK: %[[RESULT_CAST_0:.*]] = quant.scast %[[RESULT_0]] : tensor<4x!quant.uniform<{{.*}}>> to tensor<4xi8>

// CHECK: %[[RESULT_1:.*]] = "test.custom_op"()
// CHECK: %[[RESULT_CAST_1:.*]] = quant.scast %[[RESULT_1]] : tensor<?x!quant.uniform<{{.*}}>> to tensor<?xi16>

// CHECK: %[[RESULT_2:.*]] = "test.custom_op"()
// CHECK: %[[RESULT_CAST_2:.*]] = quant.scast %[[RESULT_2]] : tensor<*x!quant.uniform<{{.*}}>> to tensor<*xi8>

// CHECK: %[[RESULT_3:.*]] = "test.custom_op"()

// CHECK: return %[[RESULT_CAST_0]], %[[RESULT_CAST_1]], %[[RESULT_CAST_2]], %[[RESULT_3]]

!qalias = !quant.uniform<i8:f32, 2.0:128>
!qalias1 = !quant.uniform<i16:f32, 3.0:128>

func.func @strip_results() -> (tensor<4x!qalias>, tensor<?x!qalias1>, tensor<*x!qalias>, tensor<4xf32>) {
  %0 = "test.custom_op"() : () -> tensor<4x!qalias>
  %1 = "test.custom_op"() : () -> tensor<?x!qalias1>
  %2 = "test.custom_op"() : () -> tensor<*x!qalias>
  %3 = "test.custom_op"() : () -> tensor<4xf32>
  return %0, %1, %2, %3 : tensor<4x!qalias>, tensor<?x!qalias1>, tensor<*x!qalias>, tensor<4xf32>
}

// -----


// CHECK-LABEL: @callee
// CHECK-SAME: (tensor<4xi8>, tensor<?xi16>) -> (tensor<*xi8>, tensor<4xf32>)

// CHECK-LABEL: @strip_call

// CHECK: %[[OPERAND_0:.*]] = "test.custom_op"()
// CHECK: %[[OPERAND_0_CAST:.*]] = quant.scast %[[OPERAND_0]] : tensor<4x!quant.uniform<{{.*}}>> to tensor<4xi8>

// CHECK: %[[OPERAND_1:.*]] = "test.custom_op"()
// CHECK: %[[OPERAND_1_CAST:.*]] = quant.scast %[[OPERAND_1]] : tensor<?x!quant.uniform<{{.*}}>> to tensor<?xi16>

// CHECK: %[[RESULTS:.*]]:2 = call @callee(%[[OPERAND_0_CAST]], %[[OPERAND_1_CAST]])

// CHECK: %[[RESULT_0_CAST:.*]] = quant.scast %[[RESULTS]]#0 : tensor<*xi8> to tensor<*x!quant.uniform<{{.*}}>>
// CHECK: "test.custom_op"(%[[RESULT_0_CAST]])

// CHECK: "test.custom_op"(%[[RESULTS]]#1)

// CHECK: return

!qalias = !quant.uniform<i8:f32, 2.0:128>
!qalias1 = !quant.uniform<i16:f32, 3.0:128>

func.func private @callee(tensor<4x!qalias>, tensor<?x!qalias1>) -> (tensor<*x!qalias>, tensor<4xf32>)

func.func @strip_call() {
  %0 = "test.custom_op"() : () -> tensor<4x!qalias>
  %1 = "test.custom_op"() : () -> tensor<?x!qalias1>
  %2:2 = func.call @callee(%0, %1) : (tensor<4x!qalias>, tensor<?x!qalias1>) -> (tensor<*x!qalias>, tensor<4xf32>)
  "test.custom_op"(%2#0) : (tensor<*x!qalias>) -> ()
  "test.custom_op"(%2#1) : (tensor<4xf32>) -> ()
  return
}
