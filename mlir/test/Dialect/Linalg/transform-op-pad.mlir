// RUN: mlir-opt --transform-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

#map = affine_map<()[s0] -> (-s0 + 12, 7)>

// CHECK-LABEL: @static_sizes_output_divisible
func.func @static_sizes_output_divisible(%arg0: tensor<24x12xf32>,
                                         %arg1: tensor<12x25xf32>,
                                         %arg2: tensor<24x25xf32>,
                                         %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<24x25xf32> {
  %0 = affine.min #map()[%iv2]

  //      CHECK: %[[T0:.*]] = tensor.extract_slice %
  //      CHECK: %[[T1:.*]] = tensor.extract_slice %
  //      CHECK: %[[T2:.*]] = tensor.extract_slice %
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>

  //  CHECK-DAG: %[[CST:.*]] = arith.constant 0.

  //      CHECK: %[[T3:.*]] = tensor.pad %[[T0]] nofold
  //      CHECK: tensor.yield %[[CST]]
  //      CHECK: %[[T4:.*]] = tensor.pad %[[T1]] nofold

  //      CHECK: %[[T5:.*]] = linalg.matmul
  // CHECK-SAME:              ins(%[[T3]], %[[T4]] : tensor<4x7xf32>, tensor<7x5xf32>)
  // CHECK-SAME:              outs(%[[T2]] : tensor<4x5xf32>)

  //      CHECK: %[[T6:.*]] = tensor.extract_slice %[[T5]]
  //      CHECK: %[[T7:.*]] = bufferization.materialize_in_destination %[[T6]] in %[[T2]]
  %4 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %copy_back = transform.structured.pad %0 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.op<"bufferization.materialize_in_destination">)
    %p = transform.num_associations %copy_back : (!transform.op<"bufferization.materialize_in_destination">) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.yield
  }
}

// -----

#map = affine_map<()[s0] -> (-s0 + 12, 7)>

// CHECK-LABEL: @pad_to_multiple
func.func @pad_to_multiple(%arg0: tensor<24x12xf32>,
                           %arg1: tensor<12x25xf32>,
                           %arg2: tensor<24x25xf32>,
                           %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<24x25xf32> {
  %0 = affine.min #map()[%iv2]
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>

  //      CHECK: linalg.matmul
  // CHECK-SAME:     ins(%{{.*}}, %{{.*}} : tensor<4x7xf32>, tensor<7x6xf32>)
  // CHECK-SAME:     outs(%{{.*}} : tensor<4x6xf32>)
  %4 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %copy_back = transform.structured.pad %0 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pad_to_multiple_of=[2, 2, 1],
      pack_paddings=[1, 1, 0]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map = affine_map<()[s0] -> (-s0 + 12, 7)>

// CHECK-LABEL: @static_sizes_output_divisible_on_empty_op
func.func @static_sizes_output_divisible_on_empty_op(%arg0: tensor<24x12xf32>,
    %arg1: tensor<12x25xf32>, %arg2: tensor<24x25xf32>, %iv0: index,
    %iv1: index, %iv2: index) -> tensor<24x25xf32> {
  %0 = affine.min #map()[%iv2]

  //      CHECK: %[[T0:.*]] = tensor.empty
  //      CHECK: %[[T1:.*]] = tensor.empty
  //      CHECK: %[[T2:.*]] = tensor.empty
  %1 = tensor.empty(%0) : tensor<4x?xf32>
  %2 = tensor.empty(%0) : tensor<?x5xf32>
  %3 = tensor.empty() : tensor<4x5xf32>

  //  CHECK-DAG: %[[CST:.*]] = arith.constant 0.

  //      CHECK: %[[T3:.*]] = tensor.pad %[[T0]] nofold
  //      CHECK: tensor.yield %[[CST]]
  //      CHECK: %[[T4:.*]] = tensor.pad %[[T1]] nofold

  //      CHECK: %[[T5:.*]] = linalg.matmul
  // CHECK-SAME:              ins(%[[T3]], %[[T4]] : tensor<4x7xf32>, tensor<7x5xf32>)
  // CHECK-SAME:              outs(%[[T2]] : tensor<4x5xf32>)
  %4 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %copy_back = transform.structured.pad %0 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @pad(%arg0: tensor<24x12xf32>,
               %arg1: tensor<12x25xf32>,
               %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{op expects a padding value of type 'f32', got 0 : i32}}
    %padded, %pad, %copy_back = transform.structured.pad %0 {
      padding_values=[0: i32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @pad(%arg0: tensor<24x12xf32>,
               %arg1: tensor<12x25xf32>,
               %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expects a padding that parses to 'f32', got "{foo}"}}
    %padded, %pad, %copy_back = transform.structured.pad %0 {
      padding_values=["{foo}", 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: @pad(
func.func @pad(%arg0: tensor<24x12xf32>,
               %arg1: tensor<12x25xf32>,
               %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // This is attached to an error that is silenceable and is not reported by this transform
  //   {{when applied to this op}}
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // This error is silenceable and is not reported by this transform
    //   {{transform.structured.pad failed to apply}}
    %padded, %pad, %copy_back = transform.structured.pad %0 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Check that the padding can be applied even when the output argument of the
// linalg op is not produced by an empty op or an extract_slice op.

// CHECK-DAG: #[[$MAP_MIN:.*]] = affine_map<(d0) -> (-d0 + 2044, 16)>
// CHECK-DAG: #[[$MAP_TO_16:.*]] = affine_map<(d0) -> (-d0 + 16)>
// CHECK-LABEL: @outs_not_produced_by_empty_or_extract_slice(
// CHECK-SAME: %[[A:[^: ]*]]: tensor<128x2044xf32>,
// CHECK-SAME: %[[B:[^: ]*]]: tensor<2044x128xf32>)
func.func @outs_not_produced_by_empty_or_extract_slice(%a : tensor<128x2044xf32>, %b : tensor<2044x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>

  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c2044 = arith.constant 2044 : index
  // CHECK: scf.for %[[ARG3:.*]] = {{.*}} iter_args(%[[ARG4:.*]] = %{{.*}})
  %10 = scf.for %arg3 = %c0 to %c2044 step %c16 iter_args(%arg4 = %9) -> (tensor<128x128xf32>) {
    // CHECK: %[[MIN:.*]] = affine.min #[[$MAP_MIN]](%[[ARG3]])
    %11 = affine.min affine_map<(d0) -> (-d0 + 2044, 16)>(%arg3)
    // CHECK: %[[A_SLICE:.*]] = tensor.extract_slice %[[A]]
    // CHECK: %[[B_SLICE:.*]] = tensor.extract_slice %[[B]]
    %extracted_slice_2 = tensor.extract_slice %a[0, %arg3] [128, %11] [1, 1] : tensor<128x2044xf32> to tensor<128x?xf32>
    %extracted_slice_3 = tensor.extract_slice %b[%arg3, 0] [%11, 128] [1, 1] : tensor<2044x128xf32> to tensor<?x128xf32>
    // CHECK-DAG: %[[CST:.*]] = arith.constant 0.

    // CHECK-DAG: %[[TO_16:.*]] = affine.apply #[[$MAP_TO_16]](%[[MIN]])
    // CHECK: %[[PADDED_A_SLICE:.*]] = tensor.pad %[[A_SLICE]] nofold low[0, 0] high[0, %[[TO_16]]]
    // CHECK: tensor.yield %[[CST]]
    // CHECK: %[[PADDED_B_SLICE:.*]] = tensor.pad %[[B_SLICE]] nofold
    // The output shape is already padded, so actually we shouldn't
    // add anything to the upper bound.
    // CHECK: %[[PADDED_ARG4:.*]] = tensor.pad %[[ARG4]] nofold low[{{.*}}] high[0, 0]

    //      CHECK: %[[T5:.*]] = linalg.matmul
    // CHECK-SAME:              ins(%[[PADDED_A_SLICE]], %[[PADDED_B_SLICE]] : tensor<128x16xf32>, tensor<16x128xf32>)
    // CHECK-SAME:              outs(%[[PADDED_ARG4]] : tensor<128x128xf32>)
    %res = linalg.matmul ins(%extracted_slice_2, %extracted_slice_3 : tensor<128x?xf32>, tensor<?x128xf32>) outs(%arg4 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %res : tensor<128x128xf32>
  }
  return %10 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %copy_back = transform.structured.pad %0 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 1]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map = affine_map<()[s0] -> (-s0 + 12, 7)>

// CHECK-LABEL: @pack_everything
func.func @pack_everything(%arg0: tensor<24x12xf32>,
                           %arg1: tensor<12x25xf32>,
                           %arg2: tensor<24x25xf32>,
                           %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<24x25xf32> {
  %0 = affine.min #map()[%iv2]

  //      CHECK: %[[T0:.*]] = tensor.extract_slice %
  //      CHECK: %[[T1:.*]] = tensor.extract_slice %
  //      CHECK: %[[T2:.*]] = tensor.extract_slice %
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>

  //  CHECK-DAG: %[[CST:.*]] = arith.constant 0.

  //      CHECK: %[[PAD0:.*]] = tensor.pad %[[T0]] nofold
  //      CHECK: %[[PAD1:.*]] = tensor.pad %[[T1]] nofold
  //      CHECK: %[[PAD2:.*]] = tensor.pad %[[T2]] nofold

  //      CHECK: %[[T5:.*]] = linalg.matmul
  // CHECK-SAME:              ins(%[[PAD0]], %[[PAD1]] : tensor<4x7xf32>, tensor<7x5xf32>)
  // CHECK-SAME:              outs(%[[PAD2]] : tensor<4x5xf32>)

  // Get unpadded result (no-op in this example).
  //      CHECK: %[[T6:.*]] = tensor.extract_slice %[[T5]]
  // Copy back result to the original buffer, so that the destination of the
  // computation does not change.
  //      CHECK: %[[T7:.*]] = bufferization.materialize_in_destination %[[T6]] in %[[T2]]
  %4 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>

  //      CHECK: %[[T8:.*]] = tensor.insert_slice %[[T7]] into %{{.*}}
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %copy_back = transform.structured.pad %0 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 1]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
