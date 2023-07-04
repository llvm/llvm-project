// RUN: mlir-opt -split-input-file -verify-diagnostics \
// RUN:   -test-transform-dialect-interpreter -canonicalize \
// RUN:   -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// CHECK:       #[[$map:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK:       #[[$map1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 10)>
// CHECK-LABEL: func @tensor_pad_constant(
//  CHECK-SAME:   %[[t:.*]]: tensor<?x10xindex>, %[[l2:.*]]: index, %[[h1:.*]]: index, %[[h2:.*]]: index
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[c50:.*]] = arith.constant 50 : index
//   CHECK-DAG:   %[[dim0:.*]] = tensor.dim %[[t]], %[[c0]]
//   CHECK-DAG:   %[[size0:.*]] = affine.apply #[[$map]]()[%[[h1]], %[[dim0]]]
//   CHECK-DAG:   %[[size1:.*]] = affine.apply #[[$map1]]()[%[[l2]], %[[h2]]]
//       CHECK:   %[[alloc:.*]] = memref.alloc(%[[size0]], %[[size1]]) : memref<?x?xindex>
//       CHECK:   linalg.fill ins(%[[c50]] : index) outs(%[[alloc]] : memref<?x?xindex>)
//       CHECK:   %[[dim0:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   %[[subview:.*]] = memref.subview %[[alloc]][5, %[[l2]]] [%[[dim0]], 10] [1, 1]
//       CHECK:   memref.tensor_store %[[t]], %[[subview]]
//       CHECK:   %[[r:.*]] = bufferization.to_tensor %[[alloc]] restrict writable : memref<?x?xindex>
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[r]]
func.func @tensor_pad_constant(%t: tensor<?x10xindex>, %l2: index, %h1: index,
                               %h2: index) -> tensor<?x?xindex> {
  %0 = tensor.pad %t low[5, %l2] high[%h1, %h2] {
  ^bb0(%arg0: index, %arg1: index):
    %c = arith.constant 50 : index
    tensor.yield %c : index
  } : tensor<?x10xindex> to tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %2 = transform.structured.bufferize_to_allocation %0 : !transform.any_op
}

// -----

// CHECK-LABEL: func @tensor_pad_constant(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x10xindex>
//       CHECK:   %[[src:.*]] = bufferization.to_memref %[[t]]
//       CHECK:   %[[alloc:.*]] = memref.alloc
//       CHECK:   %[[subview:.*]] = memref.subview %[[alloc]]
//       CHECK:   memref.copy %[[src]], %[[subview]]
//       CHECK:   bufferization.to_tensor %[[alloc]] restrict writable
func.func @tensor_pad_constant(%t: tensor<?x10xindex>, %l2: index, %h1: index,
                               %h2: index) -> tensor<?x?xindex> {
  %0 = tensor.pad %t low[5, %l2] high[%h1, %h2] {
  ^bb0(%arg0: index, %arg1: index):
    %c = arith.constant 50 : index
    tensor.yield %c : index
  } : tensor<?x10xindex> to tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %2 = transform.structured.bufferize_to_allocation %0 : !transform.any_op
  // Make sure that One-Shot Bufferize can bufferize the rest.
  %4 = transform.bufferization.one_shot_bufferize %arg1 : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: func @tensor_insert(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x10xindex>
//       CHECK:   %[[m:.*]] = bufferization.to_memref %[[t]]
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}}) : memref<?x10xindex, 4>
//       CHECK:   memref.copy %[[m]], %[[alloc]]
//       CHECK:   memref.store %{{.*}}, %[[alloc]]
//       CHECK:   %[[r:.*]] = bufferization.to_tensor %[[alloc]] restrict writable
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[r]]
func.func @tensor_insert(%t: tensor<?x10xindex>, %idx: index, %v: index) -> tensor<?x10xindex> {
  %r = tensor.insert %v into %t[%idx, %idx] : tensor<?x10xindex>
  return %r : tensor<?x10xindex>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.insert"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %2 = transform.structured.bufferize_to_allocation %0 {memory_space = 4} : !transform.any_op
  // Make sure that One-Shot Bufferize can bufferize the rest.
  %4 = transform.bufferization.one_shot_bufferize %arg1 : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @tensor_extract(%t: tensor<?x10xindex>, %idx: index) -> index {
  // expected-note @below{{target payload op}}
  %r = tensor.extract %t[%idx, %idx] : tensor<?x10xindex>
  return %r : index
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.extract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below{{failed to bufferize operation}}
  %2 = transform.structured.bufferize_to_allocation %0 {memory_space = 4} : !transform.any_op
}
