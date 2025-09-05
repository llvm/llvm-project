// RUN: mlir-opt %s -test-vector-sink-patterns -split-input-file | FileCheck %s
// RUN: mlir-opt -transform-preload-library='transform-library-paths=%p/vector-sink-transform.mlir' -transform-interpreter -split-input-file %s | FileCheck %s

//-----------------------------------------------------------------------------
// [Pattern: ReorderElementwiseOpsOnBroadcast]
//-----------------------------------------------------------------------------

// CHECK-LABEL:   func.func @broadcast_scalar_with_bcast(
// CHECK-SAME:     %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index) -> vector<1x4xindex> {
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG_0]], %[[ARG_1]] : index
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADD]] : index to vector<1x4xindex>
// CHECK:           return %[[BCAST]] : vector<1x4xindex>

func.func @broadcast_scalar_with_bcast(%arg1: index, %arg2: index) -> vector<1x4xindex> {
  %0 = vector.broadcast %arg1 : index to vector<1x4xindex>
  %1 = vector.broadcast %arg2 : index to vector<1x4xindex>
  %2 = arith.addi %0, %1 : vector<1x4xindex>
  return %2 : vector<1x4xindex>
}

// CHECK-LABEL:   func.func @broadcast_scalar_with_bcast_scalable(
// CHECK-SAME:     %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index) -> vector<1x[4]xindex> {
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG_0]], %[[ARG_1]] : index
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADD]] : index to vector<1x[4]xindex>
// CHECK:           return %[[BCAST]] : vector<1x[4]xindex>

func.func @broadcast_scalar_with_bcast_scalable(%arg1: index, %arg2: index) -> vector<1x[4]xindex> {
  %0 = vector.broadcast %arg1 : index to vector<1x[4]xindex>
  %1 = vector.broadcast %arg2 : index to vector<1x[4]xindex>
  %2 = arith.addi %0, %1 : vector<1x[4]xindex>
  return %2 : vector<1x[4]xindex>
}

// -----

// CHECK-LABEL:   func.func @broadcast_scalar_with_bcast_and_splat(
// CHECK-SAME:      %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: index) -> vector<1x4xindex> {
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : index
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADD]] : index to vector<1x4xindex>
// CHECK:           return %[[BCAST]] : vector<1x4xindex>
func.func @broadcast_scalar_with_bcast_and_splat(%arg1: index, %arg2: index) -> vector<1x4xindex> {
  %0 = vector.broadcast %arg1 : index to vector<1x4xindex>
  %1 = vector.broadcast %arg2 : index to vector<1x4xindex>
  %2 = arith.addi %0, %1 : vector<1x4xindex>
  return %2 : vector<1x4xindex>
}

// CHECK-LABEL:   func.func @broadcast_scalar_with_bcast_and_splat_scalable(
// CHECK-SAME:      %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: index) -> vector<1x[4]xindex> {
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : index
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADD]] : index to vector<1x[4]xindex>
// CHECK:           return %[[BCAST]] : vector<1x[4]xindex>
func.func @broadcast_scalar_with_bcast_and_splat_scalable(%arg1: index, %arg2: index) -> vector<1x[4]xindex> {
  %0 = vector.broadcast %arg1 : index to vector<1x[4]xindex>
  %1 = vector.broadcast %arg2 : index to vector<1x[4]xindex>
  %2 = arith.addi %0, %1 : vector<1x[4]xindex>
  return %2 : vector<1x[4]xindex>
}

// -----

// CHECK-LABEL:   func.func @broadcast_vector(
// CHECK-SAME:      %[[ARG_0:.*]]: vector<4xf32>,
// CHECK-SAME:      %[[ARG_1:.*]]: vector<4xf32>) -> vector<3x4xf32> {
// CHECK:           %[[ADDF:.*]] = arith.addf %[[ARG_0]], %[[ARG_1]] : vector<4xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADDF]] : vector<4xf32> to vector<3x4xf32>
// CHECK:           return %[[BCAST]] : vector<3x4xf32>

func.func @broadcast_vector(%arg1: vector<4xf32>, %arg2: vector<4xf32>) -> vector<3x4xf32> {
  %arg1_bcast = vector.broadcast %arg1 : vector<4xf32> to vector<3x4xf32>
  %arg2_bcast = vector.broadcast %arg2 : vector<4xf32> to vector<3x4xf32>
  %2 = arith.addf %arg1_bcast, %arg2_bcast : vector<3x4xf32>
  return %2 : vector<3x4xf32>
}

// CHECK-LABEL:   func.func @broadcast_vector_scalable(
// CHECK-SAME:      %[[ARG_0:.*]]: vector<[4]xf32>,
// CHECK-SAME:      %[[ARG_1:.*]]: vector<[4]xf32>) -> vector<3x[4]xf32> {
// CHECK:           %[[ADDF:.*]] = arith.addf %[[ARG_0]], %[[ARG_1]] : vector<[4]xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADDF]] : vector<[4]xf32> to vector<3x[4]xf32>
// CHECK:           return %[[BCAST]] : vector<3x[4]xf32>

func.func @broadcast_vector_scalable(%arg1: vector<[4]xf32>, %arg2: vector<[4]xf32>) -> vector<3x[4]xf32> {
  %arg1_bcast = vector.broadcast %arg1 : vector<[4]xf32> to vector<3x[4]xf32>
  %arg2_bcast = vector.broadcast %arg2 : vector<[4]xf32> to vector<3x[4]xf32>
  %2 = arith.addf %arg1_bcast, %arg2_bcast : vector<3x[4]xf32>
  return %2 : vector<3x[4]xf32>
}

// -----

// CHECK-LABEL:   func.func @broadcast_scalar_and_vec(
// CHECK-SAME:       %[[ARG1:.*]]: index,
// CHECK-SAME:       %[[ARG2:.*]]: vector<4xindex>) -> vector<1x4xindex> {
// CHECK:            %[[SPLAT:.*]] = vector.broadcast %[[ARG1]] : index to vector<1x4xindex>
// CHECK:            %[[BCAST:.*]] = vector.broadcast %[[ARG2]] : vector<4xindex> to vector<1x4xindex>
// CHECK:            %[[ADD:.*]] = arith.addi %[[SPLAT]], %[[BCAST]] : vector<1x4xindex>
// CHECK:            return %[[ADD]] : vector<1x4xindex>
func.func @broadcast_scalar_and_vec(%arg1: index, %arg2: vector<4xindex>) -> vector<1x4xindex> {
  %0 = vector.broadcast %arg1 : index to vector<1x4xindex>
  %1 = vector.broadcast %arg2 : vector<4xindex> to vector<1x4xindex>
  %2 = arith.addi %0, %1 : vector<1x4xindex>
  return %2 : vector<1x4xindex>
}

// CHECK-LABEL:   func.func @broadcast_scalar_and_vec_scalable(
// CHECK-SAME:       %[[ARG1:.*]]: index,
// CHECK-SAME:       %[[ARG2:.*]]: vector<[4]xindex>) -> vector<1x[4]xindex> {
// CHECK:            %[[SPLAT:.*]] = vector.broadcast %[[ARG1]] : index to vector<1x[4]xindex>
// CHECK:            %[[BCAST:.*]] = vector.broadcast %[[ARG2]] : vector<[4]xindex> to vector<1x[4]xindex>
// CHECK:            %[[ADD:.*]] = arith.addi %[[SPLAT]], %[[BCAST]] : vector<1x[4]xindex>
// CHECK:            return %[[ADD]] : vector<1x[4]xindex>
func.func @broadcast_scalar_and_vec_scalable(%arg1: index, %arg2: vector<[4]xindex>) -> vector<1x[4]xindex> {
  %0 = vector.broadcast %arg1 : index to vector<1x[4]xindex>
  %1 = vector.broadcast %arg2 : vector<[4]xindex> to vector<1x[4]xindex>
  %2 = arith.addi %0, %1 : vector<1x[4]xindex>
  return %2 : vector<1x[4]xindex>
}

// -----

// CHECK-LABEL:   func.func @broadcast_vector_and_scalar(
// CHECK-SAME:      %[[ARG_0:.*]]: i32,
// CHECK-SAME:      %[[ARG_1:.*]]: vector<4xi32>) -> vector<4xi32> {
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ARG_0]] : i32 to vector<4xi32>
// CHECK:           %[[ADD:.*]] = arith.addi %[[BCAST]], %[[ARG_1]] : vector<4xi32>
// CHECK:           return %[[ADD]] : vector<4xi32>

func.func @broadcast_vector_and_scalar(%arg1: i32, %arg2: vector<4xi32>) -> vector<4xi32> {
  %arg1_bcast = vector.broadcast %arg1 : i32 to vector<4xi32>
  %2 = arith.addi %arg1_bcast, %arg2 : vector<4xi32>
  return %2 : vector<4xi32>
}

// CHECK-LABEL:   func.func @broadcast_vector_and_scalar_scalable(
// CHECK-SAME:      %[[ARG_0:.*]]: i32,
// CHECK-SAME:      %[[ARG_1:.*]]: vector<[4]xi32>) -> vector<[4]xi32> {
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ARG_0]] : i32 to vector<[4]xi32>
// CHECK:           %[[ADD:.*]] = arith.addi %[[BCAST]], %[[ARG_1]] : vector<[4]xi32>
// CHECK:           return %[[ADD]] : vector<[4]xi32>

func.func @broadcast_vector_and_scalar_scalable(%arg1: i32, %arg2: vector<[4]xi32>) -> vector<[4]xi32> {
  %arg1_bcast = vector.broadcast %arg1 : i32 to vector<[4]xi32>
  %2 = arith.addi %arg1_bcast, %arg2 : vector<[4]xi32>
  return %2 : vector<[4]xi32>
}

// -----

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL:   func.func @negative_not_elementwise
// CHECK-DAG:       %[[F1:.*]] = arith.constant dense<1.000000e+00> : vector<2x2xf32>
// CHECK-DAG:       %[[F2:.*]] = arith.constant dense<2.000000e+00> : vector<2x2xf32>
// CHECK-DAG:       %[[F3:.*]] = arith.constant dense<3.000000e+00> : vector<2x2xf32>
// CHECK:           %[[RES:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[F1]], %[[F2]], %[[F3]] : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
func.func @negative_not_elementwise() -> vector<2x2xf32> {
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32

  %A = vector.broadcast %f1 : f32 to vector<2x2xf32>
  %B = vector.broadcast %f2 : f32 to vector<2x2xf32>
  %C = vector.broadcast %f3 : f32 to vector<2x2xf32>
  %res = vector.contract #matmat_trait %A, %B, %C
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

  return %res : vector<2x2xf32>
}

// -----

// The source and the result for arith.cmp have different types

// CHECK-LABEL: func.func @source_and_result_mismatch(
//  CHECK-SAME: %[[ARG0:.+]]: f32)
//       CHECK:   %[[COMPARE:.+]] = arith.cmpf uno, %[[ARG0]], %[[ARG0]]
//       CHECK:   %[[BROADCAST:.+]] = vector.broadcast %[[COMPARE]] : i1 to vector<1xi1>
//       CHECK:   return %[[BROADCAST]]
func.func @source_and_result_mismatch(%arg0 : f32) -> vector<1xi1> {
  %0 = vector.broadcast %arg0 : f32 to vector<1xf32>
  %1 = arith.cmpf uno, %0, %0 : vector<1xf32>
  return %1 : vector<1xi1>
}

// -----

// vector.fma only supports vectors - currently it's not possible to replace this with e.g.:
//    %scalar_res = vector.fma %scalar_1, %scalar2
//    %vec_res = vector.broadcast %scalar_res
//
// TODO: It should be possible to support this case

// CHECK-LABEL: func.func @negative_op_only_supports_vectors
  //     CHECK:   %[[BROADCAST:.+]] = vector.broadcast
  //     CHECK:   %[[RESULT:.+]] = vector.fma %[[BROADCAST]]
  //     CHECK:   return %[[RESULT]]
func.func @negative_op_only_supports_vectors(%arg0 : f32) -> vector<1xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<1xf32>
  %1 = vector.fma %0, %0, %0 : vector<1xf32>
  return %1 : vector<1xf32>
}

// -----

// CHECK-LABEL:   func.func @broadcast_scalar_and_splat_const(
// CHECK-SAME:     %[[ARG_0:.*]]: index) -> vector<1x4xindex> {
// CHECK:           %[[NEW_CST:.*]] = arith.constant 2 : index
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG_0]], %[[NEW_CST]] : index
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADD]] : index to vector<1x4xindex>
// CHECK:           return %[[BCAST]] : vector<1x4xindex>

func.func @broadcast_scalar_and_splat_const(%arg0: index) -> vector<1x4xindex> {
  %0 = vector.broadcast %arg0 : index to vector<1x4xindex>
  %cst = arith.constant dense<2> : vector<1x4xindex>
  %2 = arith.addi %0, %cst : vector<1x4xindex>
  return %2 : vector<1x4xindex>
}

// -----

// CHECK-LABEL:   func.func @broadcast_scalar_and_splat_const_const_first(
// CHECK-SAME:     %[[ARG_0:.*]]: index) -> vector<1x4xindex> {
// CHECK:           %[[NEW_CST:.*]] = arith.constant 2 : index
// CHECK:           %[[SUB:.*]] = arith.subi %[[NEW_CST]], %[[ARG_0]] : index
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[SUB]] : index to vector<1x4xindex>
// CHECK:           return %[[BCAST]] : vector<1x4xindex>

func.func @broadcast_scalar_and_splat_const_const_first(%arg0: index) -> vector<1x4xindex> {
  %0 = vector.broadcast %arg0 : index to vector<1x4xindex>
  %cst = arith.constant dense<2> : vector<1x4xindex>
  %2 = arith.subi %cst, %0 : vector<1x4xindex>
  return %2 : vector<1x4xindex>
}

// -----

// CHECK-LABEL:   func.func @broadcast_vector_and_splat_const(
// CHECK-SAME:     %[[ARG_0:.*]]: vector<4xf32>) -> vector<3x4xf32> {
// CHECK:           %[[NEW_CST:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
// CHECK:           %[[ADD:.*]] = arith.mulf %[[ARG_0]], %[[NEW_CST]] : vector<4xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[ADD]] : vector<4xf32> to vector<3x4xf32>
// CHECK:           return %[[BCAST]] : vector<3x4xf32>

func.func @broadcast_vector_and_splat_const(%arg0: vector<4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf32> to vector<3x4xf32>
  %cst = arith.constant dense<2.000000e+00> : vector<3x4xf32>
  %2 = arith.mulf %0, %cst : vector<3x4xf32>
  return %2 : vector<3x4xf32>
}

// -----

// CHECK-LABEL:   func.func @negative_broadcast_with_non_splat_const(
// CHECK-SAME:     %[[ARG_0:.*]]: index) -> vector<1x4xindex> {
// CHECK-DAG:       %[[BCAST:.*]] = vector.broadcast %[[ARG_0]] : index to vector<1x4xindex>
// CHECK-DAG:       %[[CST:.*]] = arith.constant dense<{{\[}}[0, 1, 2, 3]]> : vector<1x4xindex>
// CHECK:           %[[ADD:.*]] = arith.addi %[[BCAST]], %[[CST]] : vector<1x4xindex>
// CHECK:           return %[[ADD]] : vector<1x4xindex>

func.func @negative_broadcast_with_non_splat_const(%arg0: index) -> vector<1x4xindex> {
  %0 = vector.broadcast %arg0 : index to vector<1x4xindex>
  %cst = arith.constant dense<[[0, 1, 2, 3]]> : vector<1x4xindex>
  %2 = arith.addi %0, %cst : vector<1x4xindex>
  return %2 : vector<1x4xindex>
}

// -----

// CHECK-LABEL:   func.func @broadcast_scalar_mixed_type(
// CHECK-SAME:     %[[ARG_0:.*]]: f16) -> vector<1x4xf32> {
// CHECK:           %[[EXTF:.*]] = arith.extf %[[ARG_0]] : f16 to f32
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[EXTF]] : f32 to vector<1x4xf32>
// CHECK:           return %[[BCAST]] : vector<1x4xf32>

func.func @broadcast_scalar_mixed_type(%arg0: f16) -> vector<1x4xf32> {
  %0 = vector.broadcast %arg0 : f16 to vector<1x4xf16>
  %1 = arith.extf %0 : vector<1x4xf16> to vector<1x4xf32>
  return %1 : vector<1x4xf32>
}

// -----

// CHECK-LABEL:   func.func @broadcast_vector_mixed_type(
// CHECK-SAME:     %[[ARG_0:.*]]: vector<4xf16>) -> vector<3x4xf32> {
// CHECK:           %[[EXTF:.*]] = arith.extf %[[ARG_0]] : vector<4xf16> to vector<4xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[EXTF]] : vector<4xf32> to vector<3x4xf32>
// CHECK:           return %[[BCAST]] : vector<3x4xf32>

func.func @broadcast_vector_mixed_type(%arg0: vector<4xf16>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf16> to vector<3x4xf16>
  %1 = arith.extf %0 : vector<3x4xf16> to vector<3x4xf32>
  return %1 : vector<3x4xf32>
}

// -----

// CHECK-LABEL:   func.func @broadcast_scalar_and_splat_const_mixed_type(
// CHECK-SAME:     %[[ARG_0:.*]]: f32) -> vector<1x4xf32> {
// CHECK:           %[[NEW_CST:.*]] = arith.constant 3 : i32
// CHECK:           %[[POW:.*]] = math.fpowi %[[ARG_0]], %[[NEW_CST]] : f32, i32
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[POW]] : f32 to vector<1x4xf32>
// CHECK:           return %[[BCAST]] : vector<1x4xf32>

func.func @broadcast_scalar_and_splat_const_mixed_type(%arg0: f32) -> vector<1x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<1x4xf32>
  %cst = arith.constant dense<3> : vector<1x4xi32>
  %2 = math.fpowi %0, %cst : vector<1x4xf32>, vector<1x4xi32>
  return %2 : vector<1x4xf32>
}

// -----

// CHECK-LABEL:   func.func @broadcast_vector_and_splat_const_mixed_type(
// CHECK-SAME:     %[[ARG_0:.*]]: vector<4xf32>) -> vector<3x4xf32> {
// CHECK:           %[[NEW_CST:.*]] = arith.constant dense<3> : vector<4xi32>
// CHECK:           %[[POW:.*]] = math.fpowi %[[ARG_0]], %[[NEW_CST]] : vector<4xf32>, vector<4xi32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[POW]] : vector<4xf32> to vector<3x4xf32>
// CHECK:           return %[[BCAST]] : vector<3x4xf32>

func.func @broadcast_vector_and_splat_const_mixed_type(%arg0: vector<4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<4xf32> to vector<3x4xf32>
  %cst = arith.constant dense<3> : vector<3x4xi32>
  %2 = math.fpowi %0, %cst : vector<3x4xf32>, vector<3x4xi32>
  return %2 : vector<3x4xf32>
}

//===----------------------------------------------------------------------===//
// [Pattern: ReorderCastOpsOnBroadcast]
//
// Reorder casting ops and vector ops. The casting ops have almost identical
// pattern, so only arith.extsi op is tested.
//===----------------------------------------------------------------------===//

// -----

func.func @broadcast_vector_extsi(%a : vector<4xi8>) -> vector<2x4xi32> {
  // CHECK: %[[EXT:.+]] = arith.extsi %{{.+}} : vector<4xi8> to vector<4xi32>
  // CHECK: vector.broadcast %[[EXT:.+]] : vector<4xi32> to vector<2x4xi32>
  %b = vector.broadcast %a : vector<4xi8> to vector<2x4xi8>
  %r = arith.extsi %b : vector<2x4xi8> to vector<2x4xi32>
  return %r : vector<2x4xi32>
}

// -----

func.func @broadcast_vector_extsi_scalable(%a : vector<[4]xi8>) -> vector<2x[4]xi32> {
  // CHECK: %[[EXT:.+]] = arith.extsi %{{.+}} : vector<[4]xi8> to vector<[4]xi32>
  // CHECK: vector.broadcast %[[EXT:.+]] : vector<[4]xi32> to vector<2x[4]xi32>
  %b = vector.broadcast %a : vector<[4]xi8> to vector<2x[4]xi8>
  %r = arith.extsi %b : vector<2x[4]xi8> to vector<2x[4]xi32>
  return %r : vector<2x[4]xi32>
}

// -----

func.func @broadcast_scalar_extsi(%a : i8) -> vector<2x4xi32> {
  // CHECK: %[[EXT:.+]] = arith.extsi %{{.+}} : i8 to i32
  // CHECK: vector.broadcast %[[EXT]] : i32 to vector<2x4xi32>
  %b = vector.broadcast %a : i8 to vector<2x4xi8>
  %r = arith.extsi %b : vector<2x4xi8> to vector<2x4xi32>
  return %r : vector<2x4xi32>
}

// -----

func.func @broadcast_scalar_extsi_scalable(%a : i8) -> vector<2x[4]xi32> {
  // CHECK: %[[EXT:.+]] = arith.extsi %{{.+}} : i8 to i32
  // CHECK: vector.broadcast %[[EXT]] : i32 to vector<2x[4]xi32>
  %b = vector.broadcast %a : i8 to vector<2x[4]xi8>
  %r = arith.extsi %b : vector<2x[4]xi8> to vector<2x[4]xi32>
  return %r : vector<2x[4]xi32>
}

//===----------------------------------------------------------------------===//
// [Pattern: ReorderElementwiseOpsOnTranspose]
//===----------------------------------------------------------------------===//

func.func @transpose_extsi(%a : vector<4x2xi8>) -> vector<2x4xi32> {
  // CHECK: %[[EXT:.+]] = arith.extsi %{{.+}} : vector<4x2xi8> to vector<4x2xi32>
  // CHECK: vector.transpose %[[EXT]], [1, 0] : vector<4x2xi32> to vector<2x4xi32>
  %b = vector.transpose %a, [1, 0]: vector<4x2xi8> to vector<2x4xi8>
  %r = arith.extsi %b : vector<2x4xi8> to vector<2x4xi32>
  return %r : vector<2x4xi32>
}

// -----

func.func @transpose_extsi_scalable(%a : vector<[4]x2xi8>) -> vector<2x[4]xi32> {
  // CHECK: %[[EXT:.+]] = arith.extsi %{{.+}} : vector<[4]x2xi8> to vector<[4]x2xi32>
  // CHECK: vector.transpose %[[EXT]], [1, 0] : vector<[4]x2xi32> to vector<2x[4]xi32>
  %b = vector.transpose %a, [1, 0]: vector<[4]x2xi8> to vector<2x[4]xi8>
  %r = arith.extsi %b : vector<2x[4]xi8> to vector<2x[4]xi32>
  return %r : vector<2x[4]xi32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_same_type
//  CHECK-SAME: (%[[A:.+]]: vector<4x2xf32>, %[[B:.+]]: vector<4x2xf32>)
//       CHECK:   %[[ADD:.+]] = arith.addf %[[A]], %[[B]] : vector<4x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[ADD]], [1, 0]
//       CHECK:   return %[[T]]

func.func @transpose_elementwise_same_type(%a : vector<4x2xf32>, %b : vector<4x2xf32>) -> vector<2x4xf32> {
  %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %r = arith.addf %at, %bt : vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_same_type_scalable
//  CHECK-SAME: (%[[A:.+]]: vector<[4]x2xf32>, %[[B:.+]]: vector<[4]x2xf32>)
//       CHECK:   %[[ADD:.+]] = arith.addf %[[A]], %[[B]] : vector<[4]x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[ADD]], [1, 0]
//       CHECK:   return %[[T]]

func.func @transpose_elementwise_same_type_scalable(%a : vector<[4]x2xf32>, %b : vector<[4]x2xf32>) -> vector<2x[4]xf32> {
  %at = vector.transpose %a, [1, 0]: vector<[4]x2xf32> to vector<2x[4]xf32>
  %bt = vector.transpose %b, [1, 0]: vector<[4]x2xf32> to vector<2x[4]xf32>
  %r = arith.addf %at, %bt : vector<2x[4]xf32>
  return %r : vector<2x[4]xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_diff_operand_types
//  CHECK-SAME: (%[[COND:.+]]: vector<4x2xi1>, %[[A:.+]]: vector<4x2xf32>, %[[B:.+]]: vector<4x2xf32>)
//       CHECK:   %[[S:.+]] = arith.select %[[COND]], %[[A]], %[[B]] : vector<4x2xi1>, vector<4x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[S]], [1, 0] : vector<4x2xf32> to vector<2x4xf32>
//       CHECK:   return %[[T]]
func.func @transpose_elementwise_diff_operand_types(%cond: vector<4x2xi1>, %a : vector<4x2xf32>, %b : vector<4x2xf32>) -> vector<2x4xf32> {
  %condt = vector.transpose %cond, [1, 0]: vector<4x2xi1> to vector<2x4xi1>
  %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %r = arith.select %condt, %at, %bt : vector<2x4xi1>, vector<2x4xf32>
  return %r : vector<2x4xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_diff_operand_types_scalable
//  CHECK-SAME: (%[[COND:.+]]: vector<[4]x2xi1>, %[[A:.+]]: vector<[4]x2xf32>, %[[B:.+]]: vector<[4]x2xf32>)
//       CHECK:   %[[S:.+]] = arith.select %[[COND]], %[[A]], %[[B]] : vector<[4]x2xi1>, vector<[4]x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[S]], [1, 0] : vector<[4]x2xf32> to vector<2x[4]xf32>
//       CHECK:   return %[[T]]
func.func @transpose_elementwise_diff_operand_types_scalable(%cond: vector<[4]x2xi1>, %a : vector<[4]x2xf32>, %b : vector<[4]x2xf32>) -> vector<2x[4]xf32> {
  %condt = vector.transpose %cond, [1, 0]: vector<[4]x2xi1> to vector<2x[4]xi1>
  %at = vector.transpose %a, [1, 0]: vector<[4]x2xf32> to vector<2x[4]xf32>
  %bt = vector.transpose %b, [1, 0]: vector<[4]x2xf32> to vector<2x[4]xf32>
  %r = arith.select %condt, %at, %bt : vector<2x[4]xi1>, vector<2x[4]xf32>
  return %r : vector<2x[4]xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_diff_operand_result_type
//  CHECK-SAME: (%[[A:.+]]: vector<4x2xf32>, %[[B:.+]]: vector<4x2xf32>)
//       CHECK:   %[[CMP:.+]] = arith.cmpf olt, %[[A]], %[[B]] : vector<4x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[CMP]], [1, 0] : vector<4x2xi1> to vector<2x4xi1>
//       CHECK:   return %[[T]]
func.func @transpose_elementwise_diff_operand_result_type(%a : vector<4x2xf32>, %b : vector<4x2xf32>) -> vector<2x4xi1> {
  %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
  %r = arith.cmpf olt, %at, %bt : vector<2x4xf32>
  return %r : vector<2x4xi1>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_diff_operand_result_type_scalable
//  CHECK-SAME: (%[[A:.+]]: vector<[4]x2xf32>, %[[B:.+]]: vector<[4]x2xf32>)
//       CHECK:   %[[CMP:.+]] = arith.cmpf olt, %[[A]], %[[B]] : vector<[4]x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[CMP]], [1, 0] : vector<[4]x2xi1> to vector<2x[4]xi1>
//       CHECK:   return %[[T]]
func.func @transpose_elementwise_diff_operand_result_type_scalable(%a : vector<[4]x2xf32>, %b : vector<[4]x2xf32>) -> vector<2x[4]xi1> {
  %at = vector.transpose %a, [1, 0]: vector<[4]x2xf32> to vector<2x[4]xf32>
  %bt = vector.transpose %b, [1, 0]: vector<[4]x2xf32> to vector<2x[4]xf32>
  %r = arith.cmpf olt, %at, %bt : vector<2x[4]xf32>
  return %r : vector<2x[4]xi1>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_splat_constant
//  CHECK-SAME: (%[[A:.+]]: vector<4x6x3x2xf32>)
//       CHECK:   %[[B:.+]] = arith.constant dense<5.000000e+00> : vector<4x6x3x2xf32>
//       CHECK:   %[[ADD:.+]] = arith.addf %[[A]], %[[B]] : vector<4x6x3x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[ADD]], [1, 0, 3, 2] : vector<4x6x3x2xf32> to vector<6x4x2x3xf32>
//       CHECK:   return %[[T:.+]] : vector<6x4x2x3xf32>

func.func @transpose_elementwise_splat_constant(%a : vector<4x6x3x2xf32>) -> vector<6x4x2x3xf32> {
  %b = arith.constant dense<5.0> : vector<6x4x2x3xf32>
  %at = vector.transpose %a, [1, 0, 3, 2]: vector<4x6x3x2xf32> to vector<6x4x2x3xf32>
  %r = arith.addf %at, %b : vector<6x4x2x3xf32>
  return %r : vector<6x4x2x3xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_splat_constant_scalable
//  CHECK-SAME: (%[[A:.+]]: vector<[4]x6x3x2xf32>)
//       CHECK:   %[[B:.+]] = arith.constant dense<5.000000e+00> : vector<[4]x6x3x2xf32>
//       CHECK:   %[[ADD:.+]] = arith.addf %[[A]], %[[B]] : vector<[4]x6x3x2xf32>
//       CHECK:   %[[T:.+]] = vector.transpose %[[ADD]], [1, 0, 3, 2] : vector<[4]x6x3x2xf32> to vector<6x[4]x2x3xf32>
//       CHECK:   return %[[T:.+]] : vector<6x[4]x2x3xf32>

func.func @transpose_elementwise_splat_constant_scalable(%a : vector<[4]x6x3x2xf32>) -> vector<6x[4]x2x3xf32> {
  %b = arith.constant dense<5.0> : vector<6x[4]x2x3xf32>
  %at = vector.transpose %a, [1, 0, 3, 2]: vector<[4]x6x3x2xf32> to vector<6x[4]x2x3xf32>
  %r = arith.addf %at, %b : vector<6x[4]x2x3xf32>
  return %r : vector<6x[4]x2x3xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_diff_map
//       CHECK:   vector.transpose
//       CHECK:   vector.transpose
//       CHECK:   arith.addf
func.func @transpose_elementwise_diff_map(%a : vector<4x6x3x2xf32>, %b: vector<6x2x4x3xf32>) -> vector<6x4x2x3xf32> {
  %at = vector.transpose %a, [1, 0, 3, 2]: vector<4x6x3x2xf32> to vector<6x4x2x3xf32>
  %bt = vector.transpose %b, [0, 2, 1, 3]: vector<6x2x4x3xf32> to vector<6x4x2x3xf32>
  %r = arith.addf %at, %bt : vector<6x4x2x3xf32>
  return %r : vector<6x4x2x3xf32>
}

// -----

// CHECK-LABEL: func @transpose_elementwise_diff_map_scalable
//       CHECK:   vector.transpose
//       CHECK:   vector.transpose
//       CHECK:   arith.addf
func.func @transpose_elementwise_diff_map_scalable(%a : vector<[4]x6x3x2xf32>, %b: vector<6x2x[4]x3xf32>) -> vector<6x[4]x2x3xf32> {
  %at = vector.transpose %a, [1, 0, 3, 2]: vector<[4]x6x3x2xf32> to vector<6x[4]x2x3xf32>
  %bt = vector.transpose %b, [0, 2, 1, 3]: vector<6x2x[4]x3xf32> to vector<6x[4]x2x3xf32>
  %r = arith.addf %at, %bt : vector<6x[4]x2x3xf32>
  return %r : vector<6x[4]x2x3xf32>
}

// -----

//-----------------------------------------------------------------------------
// [Pattern: ExtractOpFromElementwise]
//-----------------------------------------------------------------------------

// CHECK-LABEL: @extract_elementwise_scalar
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<4xf32>, %[[ARG1:.*]]: vector<4xf32>)
func.func @extract_elementwise_scalar(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> f32 {
// CHECK:   %[[EXT0:.*]] = vector.extract %[[ARG0]][1] : f32 from vector<4xf32>
// CHECK:   %[[EXT1:.*]] = vector.extract %[[ARG1]][1] : f32 from vector<4xf32>
// CHECK:   %[[RES:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : f32
// CHECK:   return %[[RES]] : f32
  %0 = arith.addf %arg0, %arg1 : vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_elementwise_arg_res_different_types
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<4xindex>)
func.func @extract_elementwise_arg_res_different_types(%arg0: vector<4xindex>) -> i64 {
// CHECK:   %[[EXT:.*]] = vector.extract %[[ARG0]][1] : index from vector<4xindex>
// CHECK:   %[[RES:.*]] = arith.index_cast %[[EXT]] : index to i64
// CHECK:   return %[[RES]] : i64
  %0 = arith.index_cast %arg0: vector<4xindex> to vector<4xi64>
  %1 = vector.extract %0[1] : i64 from vector<4xi64>
  return %1 : i64
}

// CHECK-LABEL: @extract_elementwise_vec
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<2x4xf32>, %[[ARG1:.*]]: vector<2x4xf32>)
func.func @extract_elementwise_vec(%arg0: vector<2x4xf32>, %arg1: vector<2x4xf32>) -> vector<4xf32> {
// CHECK:   %[[EXT0:.*]] = vector.extract %[[ARG0]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK:   %[[EXT1:.*]] = vector.extract %[[ARG1]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK:   %[[RES:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : vector<4xf32>
// CHECK:   return %[[RES]] : vector<4xf32>
  %0 = arith.addf %arg0, %arg1 : vector<2x4xf32>
  %1 = vector.extract %0[1] : vector<4xf32> from vector<2x4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @negative_extract_elementwise_no_single_use
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<4xf32>, %[[ARG1:.*]]: vector<4xf32>)
func.func @negative_extract_elementwise_no_single_use(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> (f32, vector<4xf32>) {
// Do not propagate extract, as elementwise has other uses.
// CHECK:   %[[ELT:.*]] = arith.addf %[[ARG0]], %[[ARG1]] : vector<4xf32>
// CHECK:   %[[EXT:.*]] = vector.extract %[[ELT]][1] : f32 from vector<4xf32>
// CHECK:   return %[[EXT]], %[[ELT]] : f32, vector<4xf32>
  %0 = arith.addf %arg0, %arg1 : vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1, %0 : f32, vector<4xf32>
}

// CHECK-LABEL: @negative_extract_elementwise_not_one_res
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<4xi32>, %[[ARG1:.*]]: vector<4xi32>)
func.func @negative_extract_elementwise_not_one_res(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> i32 {
// Do not propagate extract, as elementwise has more than 1 result.
// CHECK:   %[[LOW:.*]], %[[HIGH:.*]] = arith.mulsi_extended %[[ARG0]], %[[ARG1]] : vector<4xi32>
// CHECK:   %[[EXT:.*]] = vector.extract %[[LOW]][1] : i32 from vector<4xi32>
// CHECK:   return %[[EXT]] : i32
  %low, %hi = arith.mulsi_extended %arg0, %arg1 : vector<4xi32>
  %1 = vector.extract %low[1] : i32 from vector<4xi32>
  return %1 : i32
}

// CHECK-LABEL: @negative_extract_not_elementwise
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<4xi64>)
func.func @negative_extract_not_elementwise(%arg0: vector<4xi64>) -> i64 {
// `test.increment` is not an elemewise op.
// CHECK:   %[[INC:.*]] = test.increment %[[ARG0]] : vector<4xi64>
// CHECK:   %[[RES:.*]] = vector.extract %[[INC]][1] : i64 from vector<4xi64>
// CHECK:   return %[[RES]] : i64
  %0 = test.increment %arg0: vector<4xi64>
  %1 = vector.extract %0[1] : i64 from vector<4xi64>
  return %1 : i64
}

// CHECK-LABEL: @negative_extract_vec_fma
//  CHECK-SAME:   (%[[ARG0:.*]]: vector<4xf32>, %[[ARG1:.*]]: vector<4xf32>, %[[ARG2:.*]]: vector<4xf32>)
func.func @negative_extract_vec_fma(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>) -> f32 {
// `vector.fma` doesn't suppport scalars.
// CHECK:   %[[FMA:.*]] = vector.fma %[[ARG0]], %[[ARG1]], %[[ARG2]] : vector<4xf32>
// CHECK:   %[[RES:.*]] = vector.extract %[[FMA]][1] : f32 from vector<4xf32>
// CHECK:   return %[[RES]] : f32
  %0 = vector.fma %arg0, %arg1, %arg2: vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @negative_extract_dynamic_pos
func.func @negative_extract_dynamic_pos(%arg0: vector<4xf32>, %arg1 : vector<4xf32>, %idx : vector<4xindex>) -> f32 {
  // CHECK-NOT: vector.extract
  // CHECK: arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
  // CHECK: vector.extract
  // CHECK: vector.extract
  %0 = arith.addf %arg0, %arg1 : vector<4xf32>
  %1 = vector.extract %idx[0] : index from vector<4xindex>
  %2 = vector.extract %0[%1] : f32 from vector<4xf32>
  return %2 : f32
}

//-----------------------------------------------------------------------------
// [Pattern: ExtractOpFromLoad]
//-----------------------------------------------------------------------------

// CHECK-LABEL: @extract_load_scalar
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index)
func.func @extract_load_scalar(%arg0: memref<?xf32>, %arg1: index) -> f32 {
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>
// CHECK:   return %[[RES]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
  %1 = vector.extract %0[0] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_load_index
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xindex>, %[[ARG1:.*]]: index)
func.func @extract_load_index(%arg0: memref<?xindex>, %arg1: index) -> index {
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[ARG1]]] : memref<?xindex>
// CHECK:   return %[[RES]] : index
  %0 = vector.load %arg0[%arg1] : memref<?xindex>, vector<4xindex>
  %1 = vector.extract %0[0] : index from vector<4xindex>
  return %1 : index
}

// CHECK-LABEL: @extract_load_scalar_non_zero_off
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index)
func.func @extract_load_scalar_non_zero_off(%arg0: memref<?xf32>, %arg1: index) -> f32 {
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG1]], %[[C1]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[OFF]]] : memref<?xf32>
// CHECK:   return %[[RES]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_load_scalar_dyn_off
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @extract_load_scalar_dyn_off(%arg0: memref<?xf32>, %arg1: index, %arg2: index) -> f32 {
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG1]], %[[ARG2]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[OFF]]] : memref<?xf32>
// CHECK:   return %[[RES]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
  %1 = vector.extract %0[%arg2] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_load_vec_non_zero_off
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @extract_load_vec_non_zero_off(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index) -> vector<4xf32> {
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG1]], %[[C1]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[OFF]], %[[ARG2]]] : memref<?x?xf32>, vector<4xf32>
// CHECK:   return %[[RES]] : vector<4xf32>
  %0 = vector.load %arg0[%arg1, %arg2] : memref<?x?xf32>, vector<2x4xf32>
  %1 = vector.extract %0[1] : vector<4xf32> from vector<2x4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @extract_load_scalar_non_zero_off_2d_src_memref
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @extract_load_scalar_non_zero_off_2d_src_memref(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index) -> f32 {
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG2]], %[[C1]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[ARG1]], %[[OFF]]] : memref<?x?xf32>
// CHECK:   return %[[RES]] : f32
  %0 = vector.load %arg0[%arg1, %arg2] : memref<?x?xf32>, vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_load_vec_high_rank
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?x?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @extract_load_vec_high_rank(%arg0: memref<?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index) -> vector<4xf32> {
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG2]], %[[C1]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]], %[[OFF]], %[[ARG3]]] : memref<?x?x?xf32>, vector<4xf32>
// CHECK:   return %[[RES]] : vector<4xf32>
  %0 = vector.load %arg0[%arg1, %arg2, %arg3] : memref<?x?x?xf32>, vector<2x4xf32>
  %1 = vector.extract %0[1] : vector<4xf32> from vector<2x4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @negative_extract_load_scalar_from_memref_of_vec
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xvector<4xf32>>, %[[ARG1:.*]]: index)
func.func @negative_extract_load_scalar_from_memref_of_vec(%arg0: memref<?xvector<4xf32>>, %arg1: index) -> f32 {
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]]] : memref<?xvector<4xf32>>, vector<4xf32>
// CHECK:   %[[EXT:.*]] = vector.extract %[[RES]][0] : f32 from vector<4xf32>
// CHECK:   return %[[EXT]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xvector<4xf32>>, vector<4xf32>
  %1 = vector.extract %0[0] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @negative_extract_load_scalar_from_memref_of_i1
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xi1>, %[[ARG1:.*]]: index)
func.func @negative_extract_load_scalar_from_memref_of_i1(%arg0: memref<?xi1>, %arg1: index) -> i1 {
// Subbyte types are tricky, ignore them for now.
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]]] : memref<?xi1>, vector<8xi1>
// CHECK:   %[[EXT:.*]] = vector.extract %[[RES]][0] : i1 from vector<8xi1>
// CHECK:   return %[[EXT]] : i1
  %0 = vector.load %arg0[%arg1] : memref<?xi1>, vector<8xi1>
  %1 = vector.extract %0[0] : i1 from vector<8xi1>
  return %1 : i1
}

// CHECK-LABEL: @negative_extract_load_no_single_use
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index)
func.func @negative_extract_load_no_single_use(%arg0: memref<?xf32>, %arg1: index) -> (f32, vector<4xf32>) {
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>, vector<4xf32>
// CHECK:   %[[EXT:.*]] = vector.extract %[[RES]][0] : f32 from vector<4xf32>
// CHECK:   return %[[EXT]], %[[RES]] : f32, vector<4xf32>
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
  %1 = vector.extract %0[0] : f32 from vector<4xf32>
  return %1, %0 : f32, vector<4xf32>
}

// CHECK-LABEL: @negative_extract_load_scalable
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index)
func.func @negative_extract_load_scalable(%arg0: memref<?xf32>, %arg1: index) -> f32 {
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>, vector<[1]xf32>
// CHECK:   %[[EXT:.*]] = vector.extract %[[RES]][0] : f32 from vector<[1]xf32>
// CHECK:   return %[[EXT]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<[1]xf32>
  %1 = vector.extract %0[0] : f32 from vector<[1]xf32>
  return %1 : f32
}

//-----------------------------------------------------------------------------
// [Pattern: StoreOpFromSplatOrBroadcast]
//-----------------------------------------------------------------------------

// CHECK-LABEL: @store_splat
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: f32)
func.func @store_splat(%arg0: memref<?xf32>, %arg1: index, %arg2: f32) {
// CHECK:   memref.store %[[ARG2]], %[[ARG0]][%[[ARG1]]] : memref<?xf32>
  %0 = vector.broadcast %arg2 : f32 to vector<1xf32>
  vector.store %0, %arg0[%arg1] : memref<?xf32>, vector<1xf32>
  return
}

// CHECK-LABEL: @store_broadcast
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: f32)
func.func @store_broadcast(%arg0: memref<?xf32>, %arg1: index, %arg2: f32) {
// CHECK:   memref.store %[[ARG2]], %[[ARG0]][%[[ARG1]]] : memref<?xf32>
  %0 = vector.broadcast %arg2 : f32 to vector<1xf32>
  vector.store %0, %arg0[%arg1] : memref<?xf32>, vector<1xf32>
  return
}

// CHECK-LABEL: @store_broadcast_1d_to_2d
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: vector<1xf32>)
func.func @store_broadcast_1d_to_2d(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: vector<1xf32>) {
// CHECK:   vector.store %[[ARG3]], %[[ARG0]][%[[ARG1]], %[[ARG2]]] : memref<?x?xf32>, vector<1xf32>
  %0 = vector.broadcast %arg3 : vector<1xf32> to vector<1x1xf32>
  vector.store %0, %arg0[%arg1, %arg2] : memref<?x?xf32>, vector<1x1xf32>
  return
}

// CHECK-LABEL: @negative_store_scalable
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: f32)
func.func @negative_store_scalable(%arg0: memref<?xf32>, %arg1: index, %arg2: f32) {
// CHECK:   %[[RES:.*]] = vector.broadcast %[[ARG2]] : f32 to vector<[1]xf32>
// CHECK:   vector.store %[[RES]], %[[ARG0]][%[[ARG1]]] : memref<?xf32>, vector<[1]xf32>
  %0 = vector.broadcast %arg2 : f32 to vector<[1]xf32>
  vector.store %0, %arg0[%arg1] : memref<?xf32>, vector<[1]xf32>
  return
}

// CHECK-LABEL: @negative_store_memref_of_vec
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xvector<1xf32>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: f32)
func.func @negative_store_memref_of_vec(%arg0: memref<?xvector<1xf32>>, %arg1: index, %arg2: f32) {
// CHECK:   %[[RES:.*]] = vector.broadcast %[[ARG2]] : f32 to vector<1xf32>
// CHECK:   vector.store %[[RES]], %[[ARG0]][%[[ARG1]]] : memref<?xvector<1xf32>>, vector<1xf32>
  %0 = vector.broadcast %arg2 : f32 to vector<1xf32>
  vector.store %0, %arg0[%arg1] : memref<?xvector<1xf32>>, vector<1xf32>
  return
}

// CHECK-LABEL: @negative_store_more_than_one_element
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: f32)
func.func @negative_store_more_than_one_element(%arg0: memref<?xf32>, %arg1: index, %arg2: f32) {
// CHECK:   %[[RES:.*]] = vector.broadcast %[[ARG2]] : f32 to vector<4xf32>
// CHECK:   vector.store %[[RES]], %[[ARG0]][%[[ARG1]]] : memref<?xf32>, vector<4xf32>
  %0 = vector.broadcast %arg2 : f32 to vector<4xf32>
  vector.store %0, %arg0[%arg1] : memref<?xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: @negative_store_no_single_use
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: f32)
func.func @negative_store_no_single_use(%arg0: memref<?xf32>, %arg1: index, %arg2: f32) -> vector<1xf32> {
// CHECK:   %[[RES:.*]] = vector.broadcast %[[ARG2]] : f32 to vector<1xf32>
// CHECK:   vector.store %[[RES]], %[[ARG0]][%[[ARG1]]] : memref<?xf32>, vector<1xf32>
// CHECK:   return %[[RES:.*]] : vector<1xf32>
  %0 = vector.broadcast %arg2 : f32 to vector<1xf32>
  vector.store %0, %arg0[%arg1] : memref<?xf32>, vector<1xf32>
  return %0 : vector<1xf32>
}
