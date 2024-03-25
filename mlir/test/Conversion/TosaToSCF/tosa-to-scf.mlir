// RUN: mlir-opt --split-input-file --tosa-to-scf %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func @while_test
// CHECK-SAME: ([[ARG0:%.+]]: tensor<i32>)
func.func @while_test(%arg0 : tensor<i32>) -> (tensor<i32>) {
  // CHECK: [[WHILE:%.+]] = scf.while ([[ARG1:%.+]] = [[ARG0]])
  %0 = tosa.while_loop (%arg1 = %arg0) : (tensor<i32>) -> tensor<i32> {
    // CHECK: tosa.const
    %1 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>

    // CHECK: [[COMPARE:%.+]] = tosa.greater_equal
    %2 = tosa.greater_equal %1, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>

    // CHECK: [[EX:%.+]] = tensor.extract [[COMPARE]]
    // CHECK: scf.condition([[EX]]) [[ARG1]]
    tosa.yield %2 : tensor<i1>
  } do {
  // CHECK: ^bb0([[ARG1:%.+]]: tensor<i32>)
  ^bb0(%arg1: tensor<i32>):
    // CHECK: tosa.const
    %1 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>

    // CHECK: [[ADD:%.+]] = tosa.add
    %2 = tosa.add %arg1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>

    // CHECK: scf.yield [[ADD]]
    tosa.yield %2 : tensor<i32>
  }
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @if_test
// CHECK-SAME: ([[ARG0:%.+]]: tensor<f32>, [[ARG1:%.+]]: tensor<f32>, [[ARG2:%.+]]: tensor<i1>)
func.func @if_test(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> (tensor<f32>) {
  // CHECK: [[EX:%.+]] = tensor.extract [[ARG2]]
  // CHECK: [[IF:%.+]] = scf.if [[EX]] -> (tensor<f32>) {
  %0 = tosa.cond_if %arg2 -> (tensor<f32>) {

  // CHECK:   scf.yield [[ARG0]]
    tosa.yield %arg0 : tensor<f32>

  // CHECK: } else {
  } else {

  // CHECK:   scf.yield [[ARG1]]
    tosa.yield %arg1 : tensor<f32>

  // CHECK: }
  // CHECK: return [[IF]]
  }

  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @scatter_test
// CHECK-SAME: ([[VALUES_IN:%.+]]: tensor<3x7x5xi32>, [[INDICES:%.+]]: tensor<3x6xi32>, [[INPUT:%.+]]: tensor<3x6x5xi32>)
func.func @scatter_test(%values_in: tensor<3x7x5xi32>, %indices : tensor<3x6xi32>, %input: tensor<3x6x5xi32>) -> tensor<3x7x5xi32> {

  // CHECK-DAG: [[C_0:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[C_1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[C_2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[C_3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[C_5:%.+]] = arith.constant 5 : index
  // CHECK-DAG: [[C_6:%.+]] = arith.constant 6 : index
  // CHECK-DAG: [[C_0_0:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[C_1_0:%.+]] = arith.constant 1 : index
  // CHECK: [[RESULT_0:%.+]] = scf.for [[ITER_VAR_0:%.+]] = [[C_0_0]] to [[C_3]] step [[C_1_0]] iter_args([[ITER_ARG_0:%.+]] = [[VALUES_IN]]) -> (tensor<3x7x5xi32>) {
    // CHECK: [[RESULT_1:%.+]] = scf.for [[ITER_VAR_1:%.+]] = [[C_0_0]] to [[C_6]] step [[C_1_0]] iter_args([[ITER_ARG_1:%.+]] = [[ITER_ARG_0]]) -> (tensor<3x7x5xi32>) {
      // CHECK-DAG: [[EXTRACTED:%.+]] = tensor.extract [[INDICES]][[[ITER_VAR_0]], [[ITER_VAR_1]]] : tensor<3x6xi32>
      // CHECK-DAG: [[EXTRACTED_CAST:%.+]] = arith.index_cast [[EXTRACTED]] : i32 to index
      // CHECK-DAG: [[EXTRACTED_SLICE:%.+]] = tensor.extract_slice [[INPUT]][[[ITER_VAR_0]], [[ITER_VAR_1]], [[C_0_0]]] [[[C_1_0]], [[C_1_0]], [[C_5]]] [[[C_1_0]], [[C_1_0]], [[C_1_0]]] : tensor<3x6x5xi32> to tensor<?x?x?xi32>
      // CHECK-DAG: [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[EXTRACTED_SLICE]] into [[ITER_ARG_1]][[[ITER_VAR_0]], [[EXTRACTED_CAST]], [[C_0_0]]] [[[C_1_0]], [[C_1_0]], [[C_5]]] [[[C_1_0]], [[C_1_0]], [[C_1_0]]] : tensor<?x?x?xi32> into tensor<3x7x5xi32>
      // CHECK: scf.yield [[INSERTED_SLICE]] : tensor<3x7x5xi32>
    // CHECK: }
    // CHECK: scf.yield [[RESULT_1]] : tensor<3x7x5xi32>
  // CHECK: }
	%0 = "tosa.scatter"(%values_in, %indices, %input) : (tensor<3x7x5xi32>, tensor<3x6xi32>, tensor<3x6x5xi32>) -> (tensor<3x7x5xi32>)

  // CHECK: return [[RESULT_0]] : tensor<3x7x5xi32>
	return %0 : tensor<3x7x5xi32>
}
