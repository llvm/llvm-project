// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// Tests for tensor.pad
///----------------------------------------------------------------------------------------

// CHECK-LABEL: func @pad_static(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<2x?x2xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[INIT:.*]] = tensor.empty() : tensor<2x3x4xf32>
//   CHECK-DAG:   %[[VEC:.*]] = vector.broadcast %[[PAD]] : f32 to vector<2x3x4xf32>
//       CHECK:   %[[FILL:.*]] = vector.transfer_write %[[VEC]], %[[INIT]]{{.*}} : vector<2x3x4xf32>, tensor<2x3x4xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] {in_bounds = [true, false, true]} : tensor<2x?x2xf32>, vector<2x3x2xf32>
//       CHECK:   %[[RESULT:.*]] = vector.transfer_write %[[READ]], %[[FILL]][%[[C0]], %[[C0]], %[[C2]]] {in_bounds = [true, true, true]} : vector<2x3x2xf32>, tensor<2x3x4xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_static(%arg0: tensor<2x?x2xf32>, %pad_value: f32) -> tensor<2x3x4xf32> {
  %0 = tensor.pad %arg0 low[0, 0, 2] high[0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad_value : f32
    } : tensor<2x?x2xf32> to tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @pad_static_source(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<2x5x2xf32>, %[[PAD:.*]]: f32
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   %[[INIT:.*]] = tensor.empty() : tensor<2x6x4xf32>
//       CHECK:   %[[VEC:.*]] =  vector.broadcast %[[PAD]] : f32 to vector<2x6x4xf32>
//       CHECK:   %[[FILL:.*]] = vector.transfer_write %[[VEC]], %[[INIT]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<2x6x4xf32>, tensor<2x6x4xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]], %{{.*}} {in_bounds = [true, true, true]} : tensor<2x5x2xf32>, vector<2x5x2xf32>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[FILL]][%[[C0]], %[[C0]], %[[C2]]] {in_bounds = [true, true, true]} : vector<2x5x2xf32>, tensor<2x6x4xf32>
//       CHECK:   return %[[WRITE]]
func.func @pad_static_source(%arg0: tensor<2x5x2xf32>, %pad_value: f32) -> tensor<2x6x4xf32> {
  %0 = tensor.pad %arg0 low[0, 0, 2] high[0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad_value : f32
    } : tensor<2x5x2xf32> to tensor<2x6x4xf32>
  return %0 : tensor<2x6x4xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}


// -----

// CHECK-LABEL: func @pad_static_dynamic(
//  CHECK-SAME:                          %[[SRC:.*]]: tensor<1x2x2x?xf32>, %[[LOW:.*]]: index, %[[HIGH:.*]]: index
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
//       CHECK:   %[[V0:.*]] = arith.addi %[[LOW]], %[[C2]] : index
//       CHECK:   %[[V1:.*]] = arith.addi %[[V0]], %[[C3]] : index
//       CHECK:   %[[V2:.*]] = arith.addi %[[HIGH]], %[[C5]] : index
//       CHECK:   %[[DIM3:.*]] = tensor.dim %[[SRC]], %[[C3]] : tensor<1x2x2x?xf32>
//       CHECK:   %[[V4:.*]] = arith.addi %[[DIM3]], %[[C3]] : index
//       CHECK:   %[[V5:.*]] = arith.addi %[[V4]], %[[C2]] : index
//       CHECK:   %[[INIT:.*]] = tensor.empty(%[[V1]], %[[V2]], %[[V5]]) : tensor<6x?x?x?xf32>
//       CHECK:   %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[INIT]] : tensor<6x?x?x?xf32>) -> tensor<6x?x?x?xf32>
//       CHECK:   %[[SRCDIM:.*]] = tensor.dim %[[SRC]], %[[C3]] : tensor<1x2x2x?xf32>
//       CHECK:   %[[RESULT:.*]] = tensor.insert_slice %[[SRC]] into %[[FILL]][2, %[[LOW]], 3, 3] [1, 2, 2, %[[SRCDIM]]] [1, 1, 1, 1] : tensor<1x2x2x?xf32> into tensor<6x?x?x?xf32>
//       CHECK:   return %[[RESULT]]
func.func @pad_static_dynamic(%arg0: tensor<1x2x2x?xf32>, %low: index, %high: index,
                  %pad_value: f32) -> tensor<6x?x?x?xf32> {
  %0 = tensor.pad %arg0 low[2, %low, 3, 3] high[3, 3, %high, 2] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %pad_value : f32
    } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
  return %0 : tensor<6x?x?x?xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @pad_static_complex(
//   CHECK-NOT:   vector<
func.func @pad_static_complex(%arg0: tensor<2x5x2xcomplex<f32>>, %pad_value: complex<f32>) -> tensor<2x6x4xcomplex<f32>> {
  %0 = tensor.pad %arg0 low[0, 0, 2] high[0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad_value : complex<f32>
    } : tensor<2x5x2xcomplex<f32>> to tensor<2x6x4xcomplex<f32>>
  return %0 : tensor<2x6x4xcomplex<f32>>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func private @make_vector() -> tensor<12x13xf32>

// CHECK-LABEL:   func.func @pad_and_insert_slice_dest(
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<1x5x6xf32>) -> tensor<1x12x13xf32> {
// CHECK:           %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CST:.*]] = arith.constant dense<5.000000e+00> : vector<1x12x13xf32>
// CHECK:           %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:           %[[PAD_VAL:.*]] = arith.constant 5.000000e+00 : f32
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<1x12x13xf32>
// CHECK:           %[[WRITE_1:.*]] = vector.transfer_write %[[CST]], %[[EMPTY]]{{\[}}%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]] {in_bounds = [true, true, true]} : vector<1x12x13xf32>, tensor<1x12x13xf32>
// CHECK:           %[[READ_1:.*]] = vector.transfer_read %[[ARG_0]]{{\[}}%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]], %[[PAD_VAL]] {in_bounds = [true, true, true]} : tensor<1x5x6xf32>, vector<1x5x6xf32>
// CHECK:           %[[WRITE_2:.*]] = vector.transfer_write %[[READ_1]], %[[WRITE_1]]{{\[}}%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]] {in_bounds = [true, true, true]} : vector<1x5x6xf32>, tensor<1x12x13xf32>
// CHECK:           %[[MAKE_VEC:.*]] = call @make_vector() : () -> tensor<12x13xf32>
// CHECK:           %[[READ_2:.*]] = vector.transfer_read %[[MAKE_VEC]]{{\[}}%[[C0_IDX]], %[[C0_IDX]]], %[[C0]] {in_bounds = [true, true]} : tensor<12x13xf32>, vector<12x13xf32>
// CHECK:           %[[RES:.*]] = vector.transfer_write %[[READ_2]], %[[WRITE_2]]{{\[}}%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]] {in_bounds = [true, true]} : vector<12x13xf32>, tensor<1x12x13xf32>
// CHECK:           return %[[RES]] : tensor<1x12x13xf32>
func.func @pad_and_insert_slice_dest(
    %arg0: tensor<1x5x6xf32>) -> tensor<1x12x13xf32> {
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[0, 0, 0] high[0, 7, 7] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):
      tensor.yield %c5 : f32
  } : tensor<1x5x6xf32> to tensor<1x12x13xf32>
  %1 = call @make_vector() : () -> tensor<12x13xf32>
  %r = tensor.insert_slice %1 into %0[0, 0, 0][1, 12, 13][1, 1, 1] : tensor<12x13xf32> into tensor<1x12x13xf32>
  return %r : tensor<1x12x13xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4  { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @pad_tensor_non_const_pad_value
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//       CHECK:   %[[FILL:.*]] = tensor.generate
//       CHECK:     %[[RES:.*]] = arith.mulf
//       CHECK:     tensor.yield %[[RES]] : f32
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]][%[[C0]], %[[C0]]], %{{.*}} {in_bounds = [true, true]} : tensor<5x6xf32>, vector<5x6xf32>
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[FILL]][%[[C3]], %[[C4]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<12x13xf32>
//       CHECK:   return %[[WRITE]]
func.func @pad_tensor_non_const_pad_value(%arg0: tensor<5x6xf32>) -> tensor<12x13xf32> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5.0 : f32
  %0 = tensor.pad %arg0 low[3, 4] high[4, 3] {
    ^bb0(%arg1: index, %arg2: index):
      %i1 = arith.index_cast %arg1 : index to i32
      %i2 = arith.index_cast %arg2 : index to i32
      %f1 = arith.sitofp %i1 : i32 to f32
      %f2 = arith.sitofp %i2 : i32 to f32
      %m = arith.mulf %f1, %f2 : f32
      tensor.yield %m : f32
  } : tensor<5x6xf32> to tensor<12x13xf32>
  return %0 : tensor<12x13xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %3 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %5 = transform.structured.vectorize_children_and_apply_patterns %4  { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_masked_pad_static_dynamic
func.func @test_masked_pad_static_dynamic(%arg0: tensor<1x2x2x?xf32>, %low: index, %high: index,
                  %pad_value: f32) -> tensor<6x?x?x?xf32> {
  // CHECK: tensor.pad
  %0 = tensor.pad %arg0 low[2, %low, 3, 3] high[3, 3, %high, 2] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %pad_value : f32
    } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
  return %0 : tensor<6x?x?x?xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1  { vectorize_padding } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
