// RUN: mlir-opt %s -allow-unregistered-dialect -linalg-comprehensive-module-bufferize="allow-return-memref allow-unknown-ops" -split-input-file | FileCheck %s

// TODO: Bufferize result IR of bufferization.
// TODO: mlir-opt %s -allow-unregistered-dialect -linalg-comprehensive-module-bufferize="allow-return-memref allow-unknown-ops" -linalg-comprehensive-module-bufferize="allow-return-memref allow-unknown-ops" -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -allow-unregistered-dialect -linalg-comprehensive-module-bufferize="test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -allow-unregistered-dialect -linalg-comprehensive-module-bufferize="test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -allow-unregistered-dialect -linalg-comprehensive-module-bufferize="test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// CHECK-LABEL: func @use_of_unknown_op_1(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func @use_of_unknown_op_1(%t1: tensor<?xf32> {linalg.inplaceable = true})
    -> vector<5xf32> {
  // ToTensorOp is generated because the function is bufferized and has a
  // memref block argument.
  // CHECK: %[[m1_tensor:.*]] = bufferization.to_tensor %[[m1]]
  // CHECK: %[[dummy:.*]] = "test.dummy_op"(%[[m1_tensor]])
  %0 = "test.dummy_op"(%t1) : (tensor<?xf32>) -> tensor<?xf32>

  %idx = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: %[[dummy_memref:.*]] = bufferization.to_memref %[[dummy]]
  // CHECK: vector.transfer_read %[[dummy_memref]]
  %1 = vector.transfer_read %0[%idx], %cst : tensor<?xf32>, vector<5xf32>
  return %1 : vector<5xf32>
}

// -----

// CHECK-LABEL: func @use_of_unknown_op_2(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func @use_of_unknown_op_2(%t1: tensor<?xf32> {linalg.inplaceable = true})
    -> tensor<?xf32> {
  // CHECK: %[[m1_tensor:.*]] = bufferization.to_tensor %[[m1]]

  // CHECK: %[[dummy1:.*]] = "test.dummy_op"(%[[m1_tensor]])
  %0 = "test.dummy_op"(%t1) : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[dummy2:.*]] = "test.another_dummy_op"(%[[dummy1]])
  %1 = "test.another_dummy_op"(%0) : (tensor<?xf32>) -> tensor<?xf32>

  // CHECK: %[[dummy2_memref:.*]] = bufferization.to_memref %[[dummy2]]
  // CHECK: return %[[dummy2_memref]]
  return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @use_of_unknown_op_3(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func @use_of_unknown_op_3(%t1: tensor<?xf32> {linalg.inplaceable = true})
    -> (vector<5xf32>, vector<5xf32>) {
  %idx = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: %[[v1:.*]] = vector.transfer_read %[[m1]]
  %1 = vector.transfer_read %t1[%idx], %cst : tensor<?xf32>, vector<5xf32>

  // CHECK: %[[m1_tensor:.*]] = bufferization.to_tensor %[[m1]]
  // CHECK: %[[dummy:.*]] = "test.dummy_op"(%[[m1_tensor]])
  %0 = "test.dummy_op"(%t1) : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[dummy_memref:.*]] = bufferization.to_memref %[[dummy]]
  // CHECK: %[[v2:.*]] = vector.transfer_read %[[dummy_memref]]
  %2 = vector.transfer_read %0[%idx], %cst : tensor<?xf32>, vector<5xf32>

  // CHECK: return %[[v1]], %[[v2]]
  return %1, %2 : vector<5xf32>, vector<5xf32>
}

// -----

// CHECK-LABEL: func @use_of_unknown_op_4(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func @use_of_unknown_op_4(%t1: tensor<?xf32> {linalg.inplaceable = true})
    -> (vector<5xf32>, tensor<?xf32>) {
  %idx = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

  // CHECK: %[[m1_tensor:.*]] = bufferization.to_tensor %[[m1]]
  // CHECK: %[[dummy:.*]] = "test.dummy_op"(%[[m1_tensor]])
  %0 = "test.dummy_op"(%t1) : (tensor<?xf32>) -> tensor<?xf32>

  // CHECK: %[[dummy_memref:.*]] = bufferization.to_memref %[[dummy]]
  // CHECK: %[[v1:.*]] = vector.transfer_read %[[dummy_memref]]
  %1 = vector.transfer_read %0[%idx], %cst : tensor<?xf32>, vector<5xf32>

  // CHECK: %[[another_dummy:.*]] = "test.another_dummy_op"(%[[dummy]])
  %2 = "test.another_dummy_op"(%0) : (tensor<?xf32>) -> tensor<?xf32>

  // CHECK: %[[another_dummy_memref:.*]] = bufferization.to_memref %[[another_dummy]]
  // CHECK: return %[[v1]], %[[another_dummy_memref]]
  return %1, %2 : vector<5xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @use_of_bufferizable_op_in_unbufferizable_op
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func @use_of_bufferizable_op_in_unbufferizable_op(
    %t1: tensor<?xf32>, %o: index, %s: index) -> (tensor<?xf32>, tensor<?xf32>) {
  // CHECK: %[[subview:.*]] = memref.subview %[[m1]]
  %0 = tensor.extract_slice %t1[%o][%s][1] : tensor<?xf32> to tensor<?xf32>
  // CHECK: %[[subview_tensor:.*]] = bufferization.to_tensor %[[subview]]
  // CHECK: %[[dummy:.*]] = "test.dummy_op"(%[[subview_tensor]])
  %1 = "test.dummy_op"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[dummy_memref:.*]] = bufferization.to_memref %[[dummy]]
  // CHECK: return %[[subview]], %[[dummy_memref]]
  return %0, %1 : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @unused_unknown_op(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func @unused_unknown_op(%t1 : tensor<?xf32>) -> vector<5xf32> {
  %idx = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: vector.transfer_read %[[m1]]
  %1 = vector.transfer_read %t1[%idx], %cst : tensor<?xf32>, vector<5xf32>

  // ToTensorOp is inserted to pass in the result of the above bufferized op.
  // CHECK: %[[m1_tensor:.*]] = bufferization.to_tensor %[[m1]]
  // CHECK: "test.dummy_op"(%[[m1_tensor]])
  "test.dummy_op"(%t1) : (tensor<?xf32>) -> ()

  return %1 : vector<5xf32>
}

// -----

// CHECK-LABEL: func @unknown_op_not_writable
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func @unknown_op_not_writable(
    %t1 : tensor<?xf32>, %v :  vector<5xf32>, %idx : index) -> tensor<?xf32> {
  // CHECK: %[[m1_tensor:.*]] = bufferization.to_tensor %[[m1]]
  // CHECK: %[[dummy:.*]] = "test.dummy_op"(%[[m1_tensor]])
  // CHECK: %[[dummy_memref:.*]] = bufferization.to_memref %[[dummy]]
  %0 = "test.dummy_op"(%t1) : (tensor<?xf32>) -> (tensor<?xf32>)

  // The result of an unknown op is not writable. Always generate a copy.
  // Note: This copy is essential for partial bufferization. Otherwise, we could
  // introducing a RaW conflict.
  // CHECK: %[[dim:.*]] = tensor.dim %[[dummy]]
  // CHECK: %[[alloc:.*]] = memref.alloc(%[[dim]])
  // CHECK: linalg.copy(%[[dummy_memref]], %[[alloc]])
  // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
  %1 = vector.transfer_write %v, %0[%idx] : vector<5xf32>, tensor<?xf32>

  // CHECK: return %[[alloc]]
  return %1 : tensor<?xf32>
}
