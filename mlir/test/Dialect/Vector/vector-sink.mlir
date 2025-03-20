// RUN: mlir-opt %s -test-vector-sink-patterns -split-input-file | FileCheck %s

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
  %0 = vector.splat %arg1 : vector<1x4xindex>
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
  %0 = vector.splat %arg1 : vector<1x[4]xindex>
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
// CHECK:            %[[SPLAT:.*]] = vector.splat %[[ARG1]] : vector<1x4xindex>
// CHECK:            %[[BCAST:.*]] = vector.broadcast %[[ARG2]] : vector<4xindex> to vector<1x4xindex>
// CHECK:            %[[ADD:.*]] = arith.addi %[[SPLAT]], %[[BCAST]] : vector<1x4xindex>
// CHECK:            return %[[ADD]] : vector<1x4xindex>
func.func @broadcast_scalar_and_vec(%arg1: index, %arg2: vector<4xindex>) -> vector<1x4xindex> {
  %0 = vector.splat %arg1 : vector<1x4xindex>
  %1 = vector.broadcast %arg2 : vector<4xindex> to vector<1x4xindex>
  %2 = arith.addi %0, %1 : vector<1x4xindex>
  return %2 : vector<1x4xindex>
}

// CHECK-LABEL:   func.func @broadcast_scalar_and_vec_scalable(
// CHECK-SAME:       %[[ARG1:.*]]: index,
// CHECK-SAME:       %[[ARG2:.*]]: vector<[4]xindex>) -> vector<1x[4]xindex> {
// CHECK:            %[[SPLAT:.*]] = vector.splat %[[ARG1]] : vector<1x[4]xindex>
// CHECK:            %[[BCAST:.*]] = vector.broadcast %[[ARG2]] : vector<[4]xindex> to vector<1x[4]xindex>
// CHECK:            %[[ADD:.*]] = arith.addi %[[SPLAT]], %[[BCAST]] : vector<1x[4]xindex>
// CHECK:            return %[[ADD]] : vector<1x[4]xindex>
func.func @broadcast_scalar_and_vec_scalable(%arg1: index, %arg2: vector<[4]xindex>) -> vector<1x[4]xindex> {
  %0 = vector.splat %arg1 : vector<1x[4]xindex>
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

// The source and the result for arith.cmp have different types - not supported

// CHECK-LABEL: func.func @negative_source_and_result_mismatch
//       CHECK:   %[[BROADCAST:.+]] = vector.broadcast
//       CHECK:   %[[RETURN:.+]] = arith.cmpf uno, %[[BROADCAST]], %[[BROADCAST]]
//       CHECK:   return %[[RETURN]]
func.func @negative_source_and_result_mismatch(%arg0 : f32, %arg1 : vector<1xf32>) -> vector<1xi1> {
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
