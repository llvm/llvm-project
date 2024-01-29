// RUN: mlir-opt -split-input-file -verify-diagnostics \
// RUN:   -transform-interpreter -canonicalize \
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
//       CHECK:   bufferization.materialize_in_destination %[[t]] in writable %[[subview]]
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %new = transform.structured.bufferize_to_allocation %0 {emit_dealloc} : !transform.any_op

    // Ensure that one linalg.fill was generated.
    %fill_op = transform.select "linalg.fill" in %new : (!transform.any_op) -> !transform.any_op
    // expected-remark @below{{1}}
    transform.test_print_number_of_associated_payload_ir_ops %fill_op : !transform.any_op

    // Ensure that one linalg.copy was generated.
    %mat = transform.select "bufferization.materialize_in_destination" in %new : (!transform.any_op) -> !transform.any_op
    // expected-remark @below{{1}}
    transform.test_print_number_of_associated_payload_ir_ops %mat : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tensor_pad_constant_with_custom_copy(
//   CHECK-NOT:   bufferization.materialize_in_destination
//   CHECK-NOT:   memref.copy
//       CHECK:   memref.alloca
//       CHECK:   linalg.copy
func.func @tensor_pad_constant_with_custom_copy(
    %t: tensor<?x10xindex>, %l2: index, %h1: index, %h2: index)
        -> tensor<?x?xindex>
{
  %0 = tensor.pad %t low[5, %l2] high[%h1, %h2] {
  ^bb0(%arg0: index, %arg1: index):
    %c = arith.constant 50 : index
    tensor.yield %c : index
  } : tensor<?x10xindex> to tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %new = transform.structured.bufferize_to_allocation %0 {memory_space = 3, alloc_op = "memref.alloca", memcpy_op = "linalg.copy", emit_dealloc}: !transform.any_op

    // Ensure that one linalg.fill was generated.
    %fill_op = transform.select "linalg.fill" in %new : (!transform.any_op) -> !transform.any_op
    // expected-remark @below{{1}}
    transform.test_print_number_of_associated_payload_ir_ops %fill_op : !transform.any_op

    // Ensure that one linalg.copy was generated.
    %linalg_copy = transform.select "linalg.copy" in %new : (!transform.any_op) -> !transform.any_op
    // expected-remark @below{{1}}
    transform.test_print_number_of_associated_payload_ir_ops %linalg_copy : !transform.any_op

    // Ensure that one memref.alloca was generated.
    %alloca = transform.select "memref.alloca" in %new : (!transform.any_op) -> !transform.any_op
    // expected-remark @below{{1}}
    transform.test_print_number_of_associated_payload_ir_ops %alloca : !transform.any_op

    // Make sure that One-Shot Bufferize can bufferize the rest.
    %4 = transform.bufferization.one_shot_bufferize %arg1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %new = transform.structured.bufferize_to_allocation %0 {emit_dealloc} : !transform.any_op
    // Make sure that One-Shot Bufferize can bufferize the rest.
    %4 = transform.bufferization.one_shot_bufferize %arg1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {
    %0 = transform.structured.match ops{["tensor.insert"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %new = transform.structured.bufferize_to_allocation %0 {memory_space = 4, emit_dealloc} : !transform.any_op
    // Make sure that One-Shot Bufferize can bufferize the rest.
    %4 = transform.bufferization.one_shot_bufferize %arg1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tensor_insert_into_empty(
//       CHECK:   %[[alloc:.*]] = memref.alloc() : memref<10xindex, 4>
//   CHECK-NOT:   memref.copy
//       CHECK:   memref.store %{{.*}}, %[[alloc]]
//       CHECK:   %[[r:.*]] = bufferization.to_tensor %[[alloc]] restrict writable
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[r]]
func.func @tensor_insert_into_empty(%idx: index, %v: index) -> tensor<10xindex> {
  %e = tensor.empty() : tensor<10xindex>
  %r = tensor.insert %v into %e[%idx] : tensor<10xindex>
  return %r : tensor<10xindex>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {
    %0 = transform.structured.match ops{["tensor.insert"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %new = transform.structured.bufferize_to_allocation %0 {memory_space = 4, emit_dealloc} : !transform.any_op
    // Make sure that One-Shot Bufferize can bufferize the rest.
    %4 = transform.bufferization.one_shot_bufferize %arg1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @tensor_extract(%t: tensor<?x10xindex>, %idx: index) -> index {
  // expected-note @below{{target payload op}}
  %r = tensor.extract %t[%idx, %idx] : tensor<?x10xindex>
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.extract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below{{failed to bufferize operation}}
    %2, %new = transform.structured.bufferize_to_allocation %0 {memory_space = 4, emit_dealloc} : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @vector_mask(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>,
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}}) : memref<?xf32, 4>
//       CHECK:   bufferization.materialize_in_destination %[[t]] in writable %[[alloc]]
//       CHECK:   vector.mask %{{.*}} { vector.transfer_write %{{.*}}, %[[alloc]]
//       CHECK:   %[[r:.*]] = bufferization.to_tensor %[[alloc]] restrict writable
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[r]]
func.func @vector_mask(%t: tensor<?xf32>, %val: vector<16xf32>, %idx: index, %m0: vector<16xi1>) -> tensor<?xf32> {
  %r = vector.mask %m0 { vector.transfer_write %val, %t[%idx] : vector<16xf32>, tensor<?xf32> } : vector<16xi1> -> tensor<?xf32>
  return %r : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["vector.mask"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %new = transform.structured.bufferize_to_allocation %0 {memory_space = 4, emit_dealloc} : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tensor_insert_destination(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x10xindex>
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}}) : memref<?x10xindex, 4>
//       CHECK:   bufferization.materialize_in_destination %[[t]] in writable %[[alloc]]
//       CHECK:   %[[t2:.*]] = bufferization.to_tensor %[[alloc]] restrict writable
//       CHECK:   %[[inserted:.*]] = tensor.insert %{{.*}} into %[[t2]]
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[inserted]]
func.func @tensor_insert_destination(%t: tensor<?x10xindex>, %idx: index, %v: index) -> tensor<?x10xindex> {
  %r = tensor.insert %v into %t[%idx, %idx] : tensor<?x10xindex>
  return %r : tensor<?x10xindex>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %new = transform.structured.bufferize_to_allocation %0 {memory_space = 4, bufferize_destination_only, emit_dealloc} : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @scf_for_destination(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x10xindex>
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}}) : memref<?x10xindex, 4>
//       CHECK:   bufferization.materialize_in_destination %[[t]] in writable %[[alloc]]
//       CHECK:   %[[t2:.*]] = bufferization.to_tensor %[[alloc]] restrict writable
//       CHECK:   %[[for:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[t2]])
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[for]]
func.func @scf_for_destination(%t: tensor<?x10xindex>, %lb: index, %ub: index, %step: index) -> tensor<?x10xindex> {
  %r = scf.for %iv = %lb to %ub step %step iter_args(%a = %t) -> tensor<?x10xindex> {
    %b = "test.foo"(%a) : (tensor<?x10xindex>) -> (tensor<?x10xindex>)
    scf.yield %b : tensor<?x10xindex>
  }
  return %r : tensor<?x10xindex>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %new = transform.structured.bufferize_to_allocation %0 {memory_space = 4, bufferize_destination_only, emit_dealloc} : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @tensor_insert_destination_no_dealloc
//   CHECK-NOT: dealloc
func.func @tensor_insert_destination_no_dealloc(%t: tensor<?x10xindex>, %idx: index, %v: index) -> tensor<?x10xindex> {
  %r = tensor.insert %v into %t[%idx, %idx] : tensor<?x10xindex>
  return %r : tensor<?x10xindex>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %new = transform.structured.bufferize_to_allocation %0 {memory_space = 4, bufferize_destination_only} : !transform.any_op
    transform.yield
  }
}
