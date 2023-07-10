// RUN: mlir-opt %s -test-transform-dialect-interpreter -canonicalize --split-input-file | FileCheck %s

// Check that we can tile softmax on tensors.
// The tiling here is 2x3.
// So the shape used in the inner loop should be 2x3x256, however since 3
// doesn't divide the second dimension (64), we should see a '?' in the shape.
// The actual size, used through extract_slice/insert_slice, should come from a
// `min(64 - current iteration index, 3)`

// CHECK: #[[$MIN_MAP:.*]] = affine_map<(d0) -> (-d0 + 64, 3)>
// CHECK-LABEL:   func.func @softmax(
// CHECK-SAME:                       %[[VAL_0:.*]]: tensor<16x64x256xf32>) -> tensor<16x64x256xf32> {
// CHECK-DAG:       %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:       %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[TENSOR_EMPTY:.*]] = tensor.empty() : tensor<16x64x256xf32>
// CHECK:           %[[VAL_7:.*]] = scf.for %[[VAL_8:.*]] = %[[C0]] to %[[C16]] step %[[C2]] iter_args(%[[VAL_9:.*]] = %[[TENSOR_EMPTY]]) -> (tensor<16x64x256xf32>) {
// CHECK:             %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[C0]] to %[[C64]] step %[[C3]] iter_args(%[[VAL_12:.*]] = %[[VAL_9]]) -> (tensor<16x64x256xf32>) {
// CHECK:               %[[VAL_13:.*]] = affine.min #[[$MIN_MAP]](%[[VAL_11]])
// CHECK:               %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_8]], %[[VAL_11]], 0] [2, %[[VAL_13]], 256] [1, 1, 1] : tensor<16x64x256xf32> to tensor<2x?x256xf32>
// CHECK:               %[[VAL_15:.*]] = tensor.extract_slice %[[VAL_12]]{{\[}}%[[VAL_8]], %[[VAL_11]], 0] [2, %[[VAL_13]], 256] [1, 1, 1] : tensor<16x64x256xf32> to tensor<2x?x256xf32>
// CHECK:               %[[VAL_16:.*]] = linalg.softmax dimension(1) ins(%[[VAL_14]] : tensor<2x?x256xf32>) outs(%[[VAL_15]] : tensor<2x?x256xf32>) -> tensor<2x?x256xf32>
// CHECK:               %[[VAL_17:.*]] = tensor.insert_slice %[[VAL_16]] into %[[VAL_12]]{{\[}}%[[VAL_8]], %[[VAL_11]], 0] [2, %[[VAL_13]], 256] [1, 1, 1] : tensor<2x?x256xf32> into tensor<16x64x256xf32>
// CHECK:               scf.yield %[[VAL_17]] : tensor<16x64x256xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_18:.*]] : tensor<16x64x256xf32>
// CHECK:           }
// CHECK:           return %[[VAL_19:.*]] : tensor<16x64x256xf32>
// CHECK:         }
func.func @softmax(%arg0: tensor<16x64x256xf32>) -> tensor<16x64x256xf32> {
  %0 = tensor.empty() : tensor<16x64x256xf32>
  %1 = linalg.softmax
         dimension(1) ins(%arg0 : tensor<16x64x256xf32>) outs(%0 : tensor<16x64x256xf32>) -> tensor<16x64x256xf32>
  return %1 : tensor<16x64x256xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:2 = transform.structured.tile %0 [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

// -----

// Test the softmax tiling interface with the tile_to_forall_op transform and
// check that it composes properly with the fuse transform.
// This should sink the linalg.generic inside the scf.forall and run that
// generic on 2x4x256 tensors (2==16/8, 4==64/16).

// CHECK: #[[$TIMES2_MAP:.*]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: #[[$TIMES4_MAP:.*]] = affine_map<(d0) -> (d0 * 4)>
// CHECK-LABEL:   func.func @softmax_tile_n_fuse(
// CHECK-SAME:                       %[[VAL_0:.*]]: tensor<16x64x256xf32>) -> tensor<16x64x256xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<16x64x256xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<16x64x256xf32>
// CHECK:           %[[VAL_4:.*]] = scf.forall (%[[VAL_5:.*]], %[[VAL_6:.*]]) in (8, 16) shared_outs(%[[VAL_7:.*]] = %[[VAL_3]]) -> (tensor<16x64x256xf32>) {
// CHECK:             %[[VAL_8:.*]] = affine.apply #[[$TIMES2_MAP]](%[[VAL_5]])
// CHECK:             %[[VAL_9:.*]] = affine.apply #[[$TIMES4_MAP]](%[[VAL_6]])
// CHECK:             %[[VAL_10:.*]] = affine.apply #[[$TIMES2_MAP]](%[[VAL_5]])
// CHECK:             %[[VAL_11:.*]] = affine.apply #[[$TIMES4_MAP]](%[[VAL_6]])
// CHECK:             %[[VAL_12:.*]] = affine.apply #[[$TIMES2_MAP]](%[[VAL_5]])
// CHECK:             %[[VAL_13:.*]] = affine.apply #[[$TIMES4_MAP]](%[[VAL_6]])
// CHECK:             %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_11]], 0] [2, 4, 256] [1, 1, 1] : tensor<16x64x256xf32> to tensor<2x4x256xf32>
// CHECK:             %[[VAL_15:.*]] = tensor.extract_slice %[[VAL_2]]{{\[}}%[[VAL_12]], %[[VAL_13]], 0] [2, 4, 256] [1, 1, 1] : tensor<16x64x256xf32> to tensor<2x4x256xf32>
// CHECK:             %[[VAL_16:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_14]] : tensor<2x4x256xf32>) outs(%[[VAL_15]] : tensor<2x4x256xf32>) {
// CHECK:             ^bb0(%[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
// CHECK:               %[[VAL_19:.*]] = arith.addf %[[VAL_18]], %[[VAL_1]] : f32
// CHECK:               linalg.yield %[[VAL_19]] : f32
// CHECK:             } -> tensor<2x4x256xf32>
// CHECK:             %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_7]]{{\[}}%[[VAL_8]], %[[VAL_9]], 0] [2, 4, 256] [1, 1, 1] : tensor<16x64x256xf32> to tensor<2x4x256xf32>
// CHECK:             %[[VAL_21:.*]] = linalg.softmax dimension(1) ins(%[[VAL_22:.*]] : tensor<2x4x256xf32>) outs(%[[VAL_20]] : tensor<2x4x256xf32>) -> tensor<2x4x256xf32>
// CHECK:             scf.forall.in_parallel {
// CHECK:               tensor.parallel_insert_slice %[[VAL_21]] into %[[VAL_7]]{{\[}}%[[VAL_8]], %[[VAL_9]], 0] [2, 4, 256] [1, 1, 1] : tensor<2x4x256xf32> into tensor<16x64x256xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_23:.*]] : tensor<16x64x256xf32>
// CHECK:         }

func.func @softmax_tile_n_fuse(%arg0: tensor<16x64x256xf32>) -> tensor<16x64x256xf32> {
  %empty = tensor.empty() : tensor<16x64x256xf32>
  %cst = arith.constant 1.000000e+00 : f32
  %eltwise = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
       iterator_types = ["parallel", "parallel", "parallel"]
      }
      ins(%arg0 : tensor<16x64x256xf32>)
      outs(%empty : tensor<16x64x256xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %arg3Plus1 = arith.addf %arg3, %cst : f32
      linalg.yield %arg3Plus1 : f32
    } -> tensor<16x64x256xf32>

  %0 = tensor.empty() : tensor<16x64x256xf32>
  %1 = linalg.softmax
         dimension(1) ins(%eltwise : tensor<16x64x256xf32>) outs(%0 : tensor<16x64x256xf32>) -> tensor<16x64x256xf32>
  return %1 : tensor<16x64x256xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op

  // Tile the root.
  %forall_op, %tiled_op = transform.structured.tile_to_forall_op %0 num_threads [8, 16]
       : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Fuse all producers.
  %1 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.fuse_into_containing_op %1 into %forall_op
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
}
// -----

// Same as the previous test but on memrefs.

// CHECK: #[[$MIN_MAP:.*]] = affine_map<(d0) -> (-d0 + 64, 3)>
// CHECK-LABEL:   func.func @softmax_memref(
// CHECK-SAME:                              %[[VAL_0:.*]]: memref<16x64x256xf32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: memref<16x64x256xf32>) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:       %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[C3:.*]] = arith.constant 3 : index
// CHECK:           scf.for %[[VAL_7:.*]] = %[[C0]] to %[[C16]] step %[[C2]] {
// CHECK:             scf.for %[[VAL_8:.*]] = %[[C0]] to %[[C64]] step %[[C3]] {
// CHECK:               %[[VAL_9:.*]] = affine.min #[[$MIN_MAP]](%[[VAL_8]])
// CHECK:               %[[VAL_10:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_8]], 0] [2, %[[VAL_9]], 256] [1, 1, 1] : memref<16x64x256xf32> to memref<2x?x256xf32, strided<[16384, 256, 1], offset: ?>>
// CHECK:               %[[VAL_11:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_7]], %[[VAL_8]], 0] [2, %[[VAL_9]], 256] [1, 1, 1] : memref<16x64x256xf32> to memref<2x?x256xf32, strided<[16384, 256, 1], offset: ?>>
// CHECK:               linalg.softmax dimension(1) ins(%[[VAL_10]] : memref<2x?x256xf32, strided<[16384, 256, 1], offset: ?>>) outs(%[[VAL_11]] : memref<2x?x256xf32, strided<[16384, 256, 1], offset: ?>>)
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @softmax_memref(%arg0: memref<16x64x256xf32>, %arg1: memref<16x64x256xf32>) {
  linalg.softmax
    dimension(1) ins(%arg0 : memref<16x64x256xf32>) outs(%arg1 : memref<16x64x256xf32>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop:2 = transform.structured.tile %0 [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}
