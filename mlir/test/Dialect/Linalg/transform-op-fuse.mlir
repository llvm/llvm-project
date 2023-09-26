// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @fuse_unary
func.func @fuse_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[RES:.*]] = scf.for
  //     CHECK:    scf.for
  //     CHECK:       linalg.elemwise_unary
  //     CHECK:       linalg.elemwise_binary
  //     CHECK: return %[[RES]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [32, 32], tile_interchange = [0, 1]}
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

// -----

// CHECK-LABEL: func.func @fuse_unary
func.func @fuse_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //     CHECK: %[[PARTIAL_RES:.*]] = scf.for
  //     CHECK:     scf.for
  //     CHECK:       linalg.elemwise_unary
  //     CHECK:       linalg.elemwise_binary
  //     CHECK: %[[RES:.*]] = scf.for {{.*}}%[[PARTIAL_RES]]
  //     CHECK:     scf.for
  //     CHECK:       linalg.elemwise_unary
  //     CHECK:       linalg.elemwise_binary
  //     CHECK: return %[[RES]]
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.elemwise_binary ins(%0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [32, 32], tile_interchange = [0, 1]}
    : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.any_op)
  transform.loop.peel %loops#0 : (!transform.op<"scf.for">) -> !transform.any_op
}

// -----

// CHECK-LABEL: func.func @interchange_reduction
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<12x7x25xf32>)
func.func @interchange_reduction(%input: tensor<12x7x25xf32>) -> tensor<12x25xf32> {
  %five = arith.constant 5.0 : f32
  %init = tensor.empty() : tensor<12x25xf32>

//   CHECK-DAG: %[[INIT:.+]] = tensor.empty()
//   CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG: %[[C7:.+]] = arith.constant 7 : index
//   CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
//       CHECK: %[[RES:.*]] = scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step %[[C5]] iter_args(%[[FOR_ARG0:.+]] = %[[INIT]])
//       CHECK:   scf.for %[[IV1:.+]] = %{{.+}} to %{{.+}} step %[[C7]] iter_args(%[[FOR_ARG1:.+]] = %[[FOR_ARG0]])
//       CHECK:     %[[OUT_SLICE0:.+]] = tensor.extract_slice %[[INPUT]][%[[IV0]], 0, %[[IV1]]]
//       CHECK:     %[[OUT_SLICE1:.+]] = tensor.extract_slice %[[FOR_ARG1]][%[[IV0]], %[[IV1]]]
//       CHECK:     %[[FILL:.+]] = linalg.fill {{.+}} outs(%[[OUT_SLICE1]] : tensor<?x?xf32>)
//       CHECK:     scf.for %[[IV2:.+]] = %{{.+}} to %{{.+}} step %[[C4]] iter_args(%[[FOR_ARG2:.+]] = %[[FILL]])
//       CHECK:       %[[IN_SLICE:.+]] = tensor.extract_slice %[[OUT_SLICE0]]
//       CHECK:       %[[OUT_SLICE2:.+]] = tensor.extract_slice %[[FOR_ARG2]][0, 0]
//       CHECK:       linalg.generic {{.+}} ins(%[[IN_SLICE]] : tensor<?x?x?xf32>) outs(%[[OUT_SLICE2]] : tensor<?x?xf32>)
//       CHECK: return %[[RES]]

  %fill = linalg.fill ins(%five : f32) outs(%init : tensor<12x25xf32>) -> tensor<12x25xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>],
    iterator_types = ["parallel", "reduction", "parallel"]
  } ins(%input : tensor<12x7x25xf32>) outs(%fill : tensor<12x25xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %2 = arith.addf %arg0, %arg1 : f32
    linalg.yield %2 : f32
  } -> tensor<12x25xf32>
  func.return %0 : tensor<12x25xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [5, 0, 7], tile_interchange = [0, 2, 1]}
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  %2, %loops_2 = transform.structured.tile_using_for %1 [0, 4]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}

// -----

// CHECK-LABEL: func.func @unpack_elemwise
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             tensor.unpack
// CHECK:             linalg.elemwise_unary
// CHECK:         return %[[RES]]
func.func @unpack_elemwise(%arg0: tensor<16x48x8x8xf32>, %arg1: tensor<128x384xf32>) -> tensor<128x384xf32> {
  %0 = tensor.empty() : tensor<128x384xf32>
  %1 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %0
      : tensor<16x48x8x8xf32> -> tensor<128x384xf32>
  %2 = linalg.elemwise_unary ins(%1: tensor<128x384xf32>)
                             outs(%arg1: tensor<128x384xf32>) -> tensor<128x384xf32>
  return %2 : tensor<128x384xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [16, 32], tile_interchange = [0, 1]}
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

// -----

// CHECK-LABEL: func.func @pack_elemwise
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             tensor.pack
// CHECK:             linalg.elemwise_unary
// CHECK:         return %[[RES]]
func.func @pack_elemwise(%arg0: tensor<128x384xf32>, %arg1: tensor<16x48x8x8xf32>) -> tensor<16x48x8x8xf32> {
  %0 = tensor.empty() : tensor<16x48x8x8xf32>
  %1 = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %0
      : tensor<128x384xf32> -> tensor<16x48x8x8xf32>
  %2 = linalg.elemwise_unary ins(%1: tensor<16x48x8x8xf32>)
                             outs(%arg1: tensor<16x48x8x8xf32>) -> tensor<16x48x8x8xf32>
  return %2 : tensor<16x48x8x8xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %loops:2 = transform.structured.fuse %0 {tile_sizes = [3, 5, 0, 0]}
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

// -----

// CHECK-LABEL: func.func @nofuse_pack_elemwise
// CHECK:         tensor.pack
// CHECK:         %[[RES:.*]] = scf.for
// CHECK:           scf.for
// CHECK:             linalg.elemwise_unary
// CHECK:         return %[[RES]]
func.func @nofuse_pack_elemwise(%arg0: tensor<128x384xf32>, %arg1: tensor<16x48x8x8xf32>) -> tensor<16x48x8x8xf32> {
  %0 = tensor.empty() : tensor<16x48x8x8xf32>
  %1 = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %0
      : tensor<128x384xf32> -> tensor<16x48x8x8xf32>
  %2 = linalg.elemwise_unary ins(%1: tensor<16x48x8x8xf32>)
                             outs(%arg1: tensor<16x48x8x8xf32>) -> tensor<16x48x8x8xf32>
  return %2 : tensor<16x48x8x8xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %loops:3 = transform.structured.fuse %0 {tile_sizes = [3, 5, 2, 0]}
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}
