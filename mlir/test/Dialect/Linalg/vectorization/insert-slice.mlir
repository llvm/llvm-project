// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// Tests for tensor.insert_slice
///----------------------------------------------------------------------------------------

func.func private @insert_slice_static_sizes(%source: tensor<?x3x?x1xi32>) -> tensor<5x3xi32> {
  %c2 = arith.constant 2 : index
  %init = tensor.empty() : tensor<5x3xi32>

  %source_slice = tensor.extract_slice %source[0, %c2, 0, 0] [1, 1, 5, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<5x1xi32>
  %res = tensor.insert_slice %source_slice into %init[0, %c2] [5, 1] [1, 1] : tensor<5x1xi32> into tensor<5x3xi32>

  return %res : tensor<5x3xi32>
}

// CHECK-LABEL:   func.func private @insert_slice_static_sizes(
// CHECK-SAME:      %[[SEC:.*]]: tensor<?x3x?x1xi32>) -> tensor<5x3xi32> {
// CHECK:           %[[C_2:.*]] = arith.constant 2 : index
// CHECK:           %[[INIT:.*]] = tensor.empty() : tensor<5x3xi32>
// CHECK:           %[[SRC_SLICE:.*]] = tensor.extract_slice %[[SEC]][0, %[[C_2]], 0, 0] [1, 1, 5, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<5x1xi32>
// CHECK-DAG:       %[[PAD:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C_0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C_0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C_5:.*]] = arith.constant 5 : index
// CHECK-DAG:       %[[C_1:.*]] = arith.constant 1 : index
// CHECK:           %[[MASK_READ:.*]] = vector.create_mask %[[C_5]], %[[C_1]] : vector<8x1xi1>
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK_READ]] { vector.transfer_read %[[SRC_SLICE]][%[[C_0]], %[[C_0]]], %[[PAD]] {{.*}} : tensor<5x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>
// CHECK-DAG:       %[[C_0_2:.*]] = arith.constant 0 : index
// CHECK:           %[[C_5_1:.*]] = arith.constant 5 : index
// CHECK:           %[[C_1_1:.*]] = arith.constant 1 : index
// CHECK:           %[[MASK_WRITE:.*]] = vector.create_mask %[[C_5_1]], %[[C_1_1]] : vector<8x1xi1>
// CHECK:           %[[RES:.*]] = vector.mask %[[MASK_WRITE]] { vector.transfer_write %[[READ]], %[[INIT]][%[[C_0_2]], %[[C_2]]]  {in_bounds = [true, true]} : vector<8x1xi32>, tensor<5x3xi32> } : vector<8x1xi1> -> tensor<5x3xi32>
// CHECK:           return %[[RES]] : tensor<5x3xi32>



module attributes {transform.with_named_sequence} {
 transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
   %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
   transform.yield
 }
}

// -----

// One of the _source_ dimensions is dynamic (but _destination_ dimensions are static).

func.func private @insert_slice_dynamic_src_dim(%source: tensor<?x3x?x1xi32>, %size: index) -> tensor<5x3xi32> {
  %c2 = arith.constant 2 : index
  %init = tensor.empty() : tensor<5x3xi32>

  %source_slice = tensor.extract_slice %source[0, %c2, 0, 0] [1, 1, %size, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<?x1xi32>
  %res = tensor.insert_slice %source_slice into %init[0, %c2] [%size, 1] [1, 1] : tensor<?x1xi32> into tensor<5x3xi32>

  return %res : tensor<5x3xi32>
}

// CHECK-LABEL:   func.func private @insert_slice_dynamic_src_dim(
// CHECK-SAME:      %[[SRC:.*]]: tensor<?x3x?x1xi32>,
// CHECK-SAME:      %[[SIZE:.*]]: index) -> tensor<5x3xi32> {
// CHECK:           %[[C_2:.*]] = arith.constant 2 : index
// CHECK:           %[[INIT:.*]] = tensor.empty() : tensor<5x3xi32>
// CHECK:           %[[SRC_SLICE:.*]] = tensor.extract_slice %[[SRC]][0, %[[C_2]], 0, 0] [1, 1, %[[SIZE]], 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<?x1xi32>
// CHECK-DAG:       %[[PAD:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C_0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C_0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C_0_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[D0:.*]] = tensor.dim %[[SRC_SLICE]], %[[C_0_2]] : tensor<?x1xi32>
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[D0]], %[[C_1]] : vector<8x1xi1>
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[SRC_SLICE]][%[[C_0_1]], %[[C_0_1]]], %[[PAD]] {{.*}} : tensor<?x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>
// CHECK:           %[[C_0_2:.*]] = arith.constant 0 : index
// CHECK:           %[[C_5_2:.*]] = arith.constant 5 : index
// CHECK:           %[[C_1_1:.*]] = arith.constant 1 : index
// CHECK:           %[[MASK_WRITE:.*]] = vector.create_mask %[[C_5_2]], %[[C_1_1]] : vector<8x1xi1>
// CHECK:           %[[RES:.*]] = vector.mask %[[MASK_WRITE]] { vector.transfer_write %[[READ]], %[[INIT]][%[[C_0_2]], %[[C_2]]]  {in_bounds = [true, true]} : vector<8x1xi32>, tensor<5x3xi32> } : vector<8x1xi1> -> tensor<5x3xi32>
// CHECK:           return %[[RES]] : tensor<5x3xi32>

 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
    transform.yield
  }
 }

// -----

// One of the _destination_ dimensions is dynamic (but _source_ dimensions are static).

func.func private @insert_slice_dynamic_dest_dim(%source: tensor<?x3x?x1xi32>, %size: index) -> tensor<?x3xi32> {
  %c2 = arith.constant 2 : index
  %init = tensor.empty(%size) : tensor<?x3xi32>

  %source_slice = tensor.extract_slice %source[0, %c2, 0, 0] [1, 1, 5, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<5x1xi32>
  %res = tensor.insert_slice %source_slice into %init[0, %c2] [5, 1] [1, 1] : tensor<5x1xi32> into tensor<?x3xi32>

  return %res : tensor<?x3xi32>
}

// CHECK-LABEL:   func.func private @insert_slice_dynamic_dest_dim(
// CHECK-SAME:      %[[SRC:.*]]: tensor<?x3x?x1xi32>,
// CHECK-SAME:      %[[SIZE:.*]]: index) -> tensor<?x3xi32> {
// CHECK:           %[[C_2:.*]] = arith.constant 2 : index
// CHECK:           %[[INIT:.*]] = tensor.empty(%[[SIZE]]) : tensor<?x3xi32>
// CHECK:           %[[SRC_SLICE:.*]] = tensor.extract_slice %[[SRC]][0, %[[C_2]], 0, 0] [1, 1, 5, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<5x1xi32>
// CHECK-DAG:       %[[PAD:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C_5:.*]] = arith.constant 5 : index
// CHECK-DAG:       %[[C_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[MASK:.*]] = vector.create_mask %[[C_5]], %[[C_1]] : vector<8x1xi1>
// CHECK-DAG:       %[[C_0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C_0_1:.*]] = arith.constant 0 : index
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[SRC_SLICE]][%[[C_0_1]], %[[C_0_1]]], %[[PAD]] {{.*}} : tensor<5x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>

// CHECK:           %[[C_0_2:.*]] = arith.constant 0 : index
// CHECK:           %[[C_0_3:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM_0:.*]] = tensor.dim %[[INIT]], %[[C_0_3]] : tensor<?x3xi32>
// CHECK:           %[[C_1_1:.*]] = arith.constant 1 : index
// CHECK:           %[[MASK_WRITE:.*]] = vector.create_mask %[[DIM_0]], %[[C_1_1]] : vector<8x1xi1>

// CHECK:           %[[RES:.*]] = vector.mask %[[MASK_WRITE]] { vector.transfer_write %[[READ]], %[[INIT]][%[[C_0_2]], %[[C_2]]]  {in_bounds = [true, true]} : vector<8x1xi32>, tensor<?x3xi32> } : vector<8x1xi1> -> tensor<?x3xi32>
// CHECK:           return %[[RES]] : tensor<?x3xi32>

 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
    transform.yield
  }
 }

// -----

// At least one _source_ and one _destination_ dimensions are dynamic.

func.func private @insert_slice_dynamic_source_and_dest_dim(%source: tensor<?x3x?x1xi32>, %size: index) -> tensor<?x3xi32> {
  %c2 = arith.constant 2 : index
  %init = tensor.empty(%size) : tensor<?x3xi32>

  %source_slice = tensor.extract_slice %source[0, %c2, 0, 0] [1, 1, %size, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<?x1xi32>
  %res = tensor.insert_slice %source_slice into %init[0, %c2] [%size, 1] [1, 1] : tensor<?x1xi32> into tensor<?x3xi32>

  return %res : tensor<?x3xi32>
}

// CHECK-LABEL:   func.func private @insert_slice_dynamic_source_and_dest_dim(
// CHECK-SAME:      %[[SRC:.*]]: tensor<?x3x?x1xi32>,
// CHECK-SAME:      %[[SIZE:.*]]: index) -> tensor<?x3xi32> {
// CHECK:           %[[C_2:.*]] = arith.constant 2 : index
// CHECK:           %[[INIT:.*]] = tensor.empty(%[[SIZE]]) : tensor<?x3xi32>
// CHECK:           %[[SRC_SLICE:.*]] = tensor.extract_slice %[[SRC]][0, %[[C_2]], 0, 0] [1, 1, %[[SIZE]], 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<?x1xi32>
// CHECK-DAG:       %[[PAD:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C0_2:.*]] = arith.constant 0 : index
// CHECK:           %[[D0:.*]] = tensor.dim %[[SRC_SLICE]], %[[C0_2]] : tensor<?x1xi32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[D0]], %[[C1]] : vector<8x1xi1>
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[SRC_SLICE]][%[[C0_1]], %[[C0_1]]], %[[PAD]] {{.*}} : tensor<?x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>

// CHECK:           %[[C_0_3:.*]] = arith.constant 0 : index
// CHECK:           %[[C_0_4:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM_1:.*]] = tensor.dim %[[INIT]], %[[C_0_4]] : tensor<?x3xi32>

// CHECK:           %[[C_1_1:.*]] = arith.constant 1 : index
// CHECK:           %[[MASK_WRITE:.*]] = vector.create_mask %[[DIM_1]], %[[C_1_1]] : vector<8x1xi1>

// CHECK:           %[[RES:.*]] = vector.mask %[[MASK_WRITE]] { vector.transfer_write %[[READ]], %[[INIT]][%[[C_0_3]], %[[C_2]]]  {in_bounds = [true, true]} : vector<8x1xi32>, tensor<?x3xi32> } : vector<8x1xi1> -> tensor<?x3xi32>


 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
    transform.yield
  }
 }

// -----

// One of the destination dimensions is dynamic and the the corresponding
// insert index is != 0. Make sure that the mask is computed correctly (note arith.subi in the output).

func.func private @insert_slice_non_zero_offset_for_dyn_dim(%source: tensor<?x3x?x1xi32>, %size: index) -> tensor<?x3xi32> {
  %c2 = arith.constant 2 : index
  %init = tensor.empty(%size) : tensor<?x3xi32>

  %source_slice = tensor.extract_slice %source[0, %c2, 0, 0] [1, 1, 6, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<6x1xi32>
  %res = tensor.insert_slice %source_slice into %init[%c2, 0] [6, 1] [1, 1] : tensor<6x1xi32> into tensor<?x3xi32>

  return %res : tensor<?x3xi32>
}

// CHECK-LABEL:   func.func private @insert_slice_non_zero_offset_for_dyn_dim(
// CHECK-SAME:      %[[SRC:.*]]: tensor<?x3x?x1xi32>,
// CHECK-SAME:      %[[SIZE:.*]]: index) -> tensor<?x3xi32> {
// CHECK:           %[[CST_2:.*]] = arith.constant 2 : index

// CHECK:           %[[INIT:.*]] = tensor.empty(%[[SIZE]]) : tensor<?x3xi32>
// CHECK:           %[[SRC_SLICE:.*]] = tensor.extract_slice %[[SRC]][0, %[[CST_2]], 0, 0]

/// Read Op
// CHECK:           %[[READ:.*]] = {{.*}} vector.transfer_read %[[SRC_SLICE]]

/// Mask for the write operatoin
// CHECK:           %[[CST_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CST_1:.*]] = arith.constant 0 : index
// CHECK:           %[[OUT_DIM_0:.*]] = tensor.dim %[[INIT]], %[[CST_1]] : tensor<?x3xi32>
// CHECK:           %[[LB:.*]] = arith.subi %[[OUT_DIM_0]], %[[CST_2]] : index
// CHECK:           %[[CST_3:.*]] = arith.constant 3 : index
// CHECK:           %[[MASK_WRITE:.*]] = vector.create_mask %[[LB]], %[[CST_3]] : vector<8x1xi1>

/// Write Op
// CHECK:           vector.mask %[[MASK_WRITE]] { vector.transfer_write %[[READ]], %[[INIT]]

 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
    transform.yield
  }
 }
