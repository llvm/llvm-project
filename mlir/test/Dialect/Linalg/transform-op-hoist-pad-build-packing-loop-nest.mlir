// BUILD-PACKING-LOOP-NEST only checks the creation of packing code but does not connect it.
// Do not run canonicalization as it would be DCE'd away.
// RUN: mlir-opt --transform-interpreter -split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=BUILD-PACKING-LOOP-NEST

func.func @pad_and_hoist_rhs(
  %arg0: tensor<24x12xf32>, %arg1: tensor<12x25xf32>, %arg2: tensor<24x25xf32>)
     -> tensor<24x25xf32>
{
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op

    %matmul_l1, %loops_l1 = transform.structured.tile_using_for %matmul [5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %matmul_padded, %0, %copy_back = transform.structured.pad %matmul_l1 {
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // In this case, the pad op is actually empty: we only tile the first dimension
    // and it does not have an impact on the RHS operand.
    %pad = transform.get_producer_of_operand %matmul_padded[1]
      : (!transform.any_op) -> !transform.any_op

    // expected-error @below {{requires exactly 2 non-null handles}}
    transform.structured.hoist_pad.build_packing_loop_nest %pad above %loops_l1
       : (!transform.any_op, !transform.any_op) -> !transform.any_op
       transform.yield
  }
}

// -----

func.func @pad_and_hoist_init(
  %arg0: tensor<24x12xf32>, %arg1: tensor<12x25xf32>, %arg2: tensor<24x25xf32>)
     -> tensor<24x25xf32>
{
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op

    %matmul_l1, %loops_l1 = transform.structured.tile_using_for %matmul [5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %matmul_padded, %0, %copy_back = transform.structured.pad %matmul_l1 {
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %pad = transform.get_producer_of_operand %matmul_padded[2]
      : (!transform.any_op) -> !transform.any_op

    // We do not know yet how to hoist the init.
    // expected-error @below {{could not build packing loop nest}}
    transform.structured.hoist_pad.build_packing_loop_nest %pad above %loops_l1
       : (!transform.any_op, !transform.any_op) -> !transform.any_op
       transform.yield
  }
}

// -----

//     BUILD-PACKING-LOOP-NEST-LABEL: pad_and_hoist_lhs
func.func @pad_and_hoist_lhs(
  %arg0: tensor<24x12xf32>, %arg1: tensor<12x25xf32>, %arg2: tensor<24x25xf32>)
     -> tensor<24x25xf32>
{
  //     BUILD-PACKING-LOOP-NEST: %[[PACKED:.*]] = scf.for %{{.*}} -> (tensor<?x5x12xf32>) {
  //     BUILD-PACKING-LOOP-NEST:   tensor.pad %{{.*}} 
  //     BUILD-PACKING-LOOP-NEST:     : tensor<?x12xf32> to tensor<5x12xf32>
  //     BUILD-PACKING-LOOP-NEST:   tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0, 0] [1, 5, 12] [1, 1, 1]
  // BUILD-PACKING-LOOP-NEST-SAME:   : tensor<5x12xf32> into tensor<?x5x12xf32>
  //     BUILD-PACKING-LOOP-NEST: scf.for %{{.*}} -> (tensor<24x25xf32>)
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op

    %matmul_l1, %loops_l1 = transform.structured.tile_using_for %matmul [5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %matmul_padded, %0, %copy_back = transform.structured.pad %matmul_l1 {
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %pad = transform.get_producer_of_operand %matmul_padded[0]
      : (!transform.any_op) -> !transform.any_op

    transform.structured.hoist_pad.build_packing_loop_nest %pad above %loops_l1
       : (!transform.any_op, !transform.any_op) -> !transform.any_op
       transform.yield
  }
}

// -----

//     BUILD-PACKING-LOOP-NEST-LABEL: pad_and_hoist_lhs_transpose
func.func @pad_and_hoist_lhs_transpose(
  %arg0: tensor<24x12xf32>, %arg1: tensor<12x25xf32>, %arg2: tensor<24x25xf32>)
     -> tensor<24x25xf32>
{
  //     BUILD-PACKING-LOOP-NEST: %[[PACKED:.*]] = scf.for %{{.*}} -> (tensor<?x12x5xf32>) {
  //     BUILD-PACKING-LOOP-NEST:   tensor.pad %{{.*}}
  //     BUILD-PACKING-LOOP-NEST:     : tensor<?x12xf32> to tensor<5x12xf32>
  //     BUILD-PACKING-LOOP-NEST:   linalg.generic
  //     BUILD-PACKING-LOOP-NEST:     -> tensor<12x5xf32>
  //     BUILD-PACKING-LOOP-NEST:   tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0, 0] [1, 12, 5] [1, 1, 1]
  // BUILD-PACKING-LOOP-NEST-SAME:   : tensor<12x5xf32> into tensor<?x12x5xf32>
  //     BUILD-PACKING-LOOP-NEST: scf.for %{{.*}} -> (tensor<24x25xf32>)
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op

    %matmul_l1, %loops_l1 = transform.structured.tile_using_for %matmul [5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %matmul_padded, %0, %copy_back = transform.structured.pad %matmul_l1 {
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %pad = transform.get_producer_of_operand %matmul_padded[0]
      : (!transform.any_op) -> !transform.any_op

    transform.structured.hoist_pad.build_packing_loop_nest %pad above %loops_l1, transpose by [1, 0]
       : (!transform.any_op, !transform.any_op) -> !transform.any_op
       transform.yield
  }
}

// -----

//     BUILD-PACKING-LOOP-NEST-LABEL: pad_and_hoist_init
func.func @pad_and_hoist_init(
  %arg0: tensor<24x12xf32>, %arg1: tensor<12x25xf32>, %arg2: tensor<24x25xf32>)
     -> tensor<24x25xf32>
{

  //      BUILD-PACKING-LOOP-NEST: scf.for %{{.*}} -> (tensor<24x25xf32>) {
  //      BUILD-PACKING-LOOP-NEST:   %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice
  //      BUILD-PACKING-LOOP-NEST:   %[[PADDED:.*]] = tensor.pad %[[EXTRACTED_SLICE]]
  //      BUILD-PACKING-LOOP-NEST:     : tensor<?x25xf32> to tensor<5x25xf32>
  //      BUILD-PACKING-LOOP-NEST:   scf.for %{{.*}} iter_args({{.*}} = %[[EXTRACTED_SLICE]]) -> (tensor<24x25xf32>, tensor<?x25xf32>) {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op

    %matmul_l1, %loops_l1:2 = transform.structured.tile_using_for %matmul [5, 0, 7] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %matmul_padded, %0, %copy_back = transform.structured.pad %matmul_l1 {
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %pad = transform.get_producer_of_operand %matmul_padded[2]
      : (!transform.any_op) -> !transform.any_op

    transform.structured.hoist_pad.build_packing_loop_nest %pad above %loops_l1#1
       : (!transform.any_op, !transform.any_op) -> !transform.any_op
       transform.yield
  }
}
