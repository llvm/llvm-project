// RUN: mlir-opt --test-transform-dialect-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.tile %0 [4, 4, 4] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
}

// CHECK-LABEL: func @tile_linalg_matmul(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<128x128xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<128x128xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<128x128xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:                                   outs(%[[sTC]] : tensor<4x4xf32>)  -> tensor<4x4xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<4x4xf32> into tensor<128x128xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<128x128xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<128x128xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<128x128xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

//      CHECK: return %[[TD0]] : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = transform.structured.match ops{["func.call"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %2, %loops:3 = transform.structured.tile %0 [%1, %1, 4] : (!pdl.operation, !pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
}

func.func private @get_dynamic_tile_size() -> index

// CHECK-LABEL: func @tile_linalg_matmul_dynamic(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func.func @tile_linalg_matmul_dynamic(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<128x128xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<128x128xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<128x128xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<128x128xf32> to tensor<?x4xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<128x128xf32> to tensor<4x?xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<128x128xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<?x4xf32>, tensor<4x?xf32>)
// CHECK-SAME:                                   outs(%[[sTC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xf32> into tensor<128x128xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<128x128xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<128x128xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<128x128xf32>
  %sz = func.call @get_dynamic_tile_size() : () -> index
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

//      CHECK: return %[[TD0]] : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-note @below {{for this parameter}}
  %1 = transform.test_produce_integer_param_with_type i64 : !transform.param<i64>
  // expected-error @below {{expected as many parameter values (0) as target ops (2)}}
  transform.structured.tile %0 [%1, %1, %1]
    : (!pdl.operation, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
}

func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  %1 = linalg.matmul  ins(%0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0, %1 : tensor<128x128xf32>, tensor<128x128xf32>
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-note @below {{for this handle}}
  %1 = transform.structured.match ops{["arith.constant"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{expected as many dynamic size-producing operations (0) as target ops (2)}}
  transform.structured.tile %0 [%1, %1, 1]
    : (!pdl.operation, !pdl.operation, !pdl.operation)
    -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
}

func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  %1 = linalg.matmul  ins(%0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0, %1 : tensor<128x128xf32>, tensor<128x128xf32>
}

// -----

// CHECK-LABEL: tile_tensor_pad
func.func @tile_tensor_pad(
  %arg0 : tensor<?x?xf32>, %cst : f32, %low: index, %high: index) 
    -> tensor<20x40xf32>
{
  // CHECK: scf.forall
  // CHECK:   scf.if
  // CHECK:     tensor.generate
  // CHECK:   else
  // CHECK:     tensor.pad {{.*}} nofold 
  %0 = tensor.pad %arg0 nofold low[%low, %low] high[%high, %high] {
        ^bb0(%arg9: index, %arg10: index):
          tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<20x40xf32>
  return %0 : tensor<20x40xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.tile_to_forall_op %0 tile_sizes[1, 1]
}
