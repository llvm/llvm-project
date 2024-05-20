// RUN: mlir-opt %s --transform-interpreter --split-input-file -resolve-shaped-type-result-dims -canonicalize | FileCheck %s

// Demonstrates what happens when peeling the 4th loop (that corresponds to the
// "depth" dimension in depthwise convs) followed by vectorization in the
// presence of _scalable_ vectors (these are introduced through scalable
// tiling). The main goal is to verify that canonicalizations fold away the
// masks in the main loop.

func.func @conv(%arg0: tensor<1x1080x1962x48xi32>, %arg1: tensor<1x43x48xi32>) -> tensor<1x1080x1920x48xi32> {
// CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (-(48 mod s0) + 48)>

// CHECK-LABEL:   func.func @conv(
// CHECK-DAG:       %[[C_43:.*]] = arith.constant 43 : index
// CHECK-DAG:       %[[C_48:.*]] = arith.constant 48 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[VSCALE:.*]] = vector.vscale
// CHECK:           %[[VSCALE_X_4:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index

// Loop over the channel/depth dim - the main part after vectorisation (vectorized, no masking)
// CHECK:               %[[UB_DEPTH_LOOP:.*]] = affine.apply #[[$MAP]](){{\[}}%[[VSCALE_X_4]]]
// CHECK-NEXT:          %[[VAL_21:.*]] = scf.for {{.*}} to %[[UB_DEPTH_LOOP]] step %[[VSCALE_X_4]]
// Loop over the Filter width dim
// CHECK:                 scf.for %{{.*}} = %[[C0]] to %[[C_43]] step %[[C1]] {{.*}} -> (tensor<1x1x4x?xi32>) {
// CHECK-NOT:               vector.mask
// CHECK:                   vector.broadcast {{.*}} : vector<[4]xi32> to vector<1x4x[4]xi32>
// CHECK-NEXT:              arith.muli {{.*}} : vector<1x4x[4]xi32>
// CHECK-NEXT:              arith.addi {{.*}} : vector<1x4x[4]xi32>
// CHECK-NOT:               vector.mask
// CHECK:                   scf.yield {{.*}} : tensor<1x1x4x?xi32>
// CHECK:                 }
// CHECK:                 tensor.insert_slice {{.*}}  tensor<1x1x4x?xi32> into tensor<1x1080x1920x48xi32>
// CHECK:                 scf.yield {{.*}} : tensor<1x1080x1920x48xi32>

// CHECK-NEXT:          }

// Loop over the channel/depth dim - the remainder part (not vectorized)
// CHECK:               scf.for {{.*}} to %[[C_48]] step %[[VSCALE_X_4]]
// Loop over the Filter width dim
// CHECK:                 scf.for %{{.*}} = %[[C0]] to %[[C_43]] step %[[C1]] {{.*}} -> (tensor<1x1x4x?xi32>) {
// CHECK:                   linalg.depthwise_conv_1d_nwc_wc {{.*}} -> tensor<1x4x?xi32>
// CHECK:                   scf.yield %{{.*}} : tensor<1x1x4x?xi32>
// CHECK:                 }
// CHECK:                 tensor.insert_slice {{.*}} tensor<1x1x4x?xi32> into tensor<1x1080x1920x48xi32>
// CHECK-NEXT:            scf.yield %{{.*}} : tensor<1x1080x1920x48xi32>
// CHECK-NEXT:          }


  %0 = tensor.empty() : tensor<1x1080x1920x48xi32>
  %c0_i32 = arith.constant 0 : i32
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<1x1080x1920x48xi32>) -> tensor<1x1080x1920x48xi32>
  %2 = linalg.depthwise_conv_2d_nhwc_hwc {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x1080x1962x48xi32>, tensor<1x43x48xi32>) outs(%1 : tensor<1x1080x1920x48xi32>) -> tensor<1x1080x1920x48xi32>
  return %2 : tensor<1x1080x1920x48xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consume}) {
    // 1. Tile parallel dims
    %1 = transform.structured.match ops{["linalg.depthwise_conv_2d_nhwc_hwc"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_0, %loops_1:4 = transform.structured.tile_using_for %1 tile_sizes [1, 1, 4, [4], 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)

    // 2. Tile reduction dims
    %2 = transform.structured.match ops{["linalg.depthwise_conv_2d_nhwc_hwc"]} in %loops_1#3 : (!transform.op<"scf.for">) -> !transform.any_op
    %tiled_linalg_op_1, %loops_2:2 = transform.structured.tile_using_for %2 tile_sizes [0, 0, 0, 0, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // 3. Decompose 2D conv into 2 x 1D conv
    %3 = transform.structured.match ops{["linalg.depthwise_conv_2d_nhwc_hwc"]} in %loops_1#3 : (!transform.op<"scf.for">) -> !transform.any_op
    %4 = transform.structured.decompose %3 : (!transform.any_op) -> !transform.any_op

    // 4. Apply loop peeling - only the 4th loop
    %main_loop, %remainder_loop = transform.loop.peel %loops_1#3 : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)
    %5 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %main_loop : (!transform.op<"scf.for">) -> !transform.any_op

    // 5. Vectorize, but only the main loop
    transform.structured.vectorize %5 vector_sizes [2, 4, [4], 16] : !transform.any_op

    transform.yield
  }
}
