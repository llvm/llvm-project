// RUN: mlir-opt -outline-shape-computation -test-print-shape-mapping -split-input-file %s 2>%t | FileCheck %s
// RUN: cat %t | FileCheck %s --check-prefix SHAPE

// Two dynamic shapes: one of direct shape.shape_of(arg) and the other.
func.func @two_dynamic_one_direct_shape(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) -> tensor<?x4x?xf32> {
  // SHAPE-DAG: Shape for {{.*}} = "test.abs"({{.*}}> :: @shape_cal_0(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
  // SHAPE-DAG: Shape for {{.*}} = "test.concat"({{.*}}> :: @shape_cal_1(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %0 = shape.shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
  %1 = shape.get_extent %0, %c2 : tensor<3xindex>, index -> index
  %2 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
  %3 = shape.with_shape %2, %0 : tensor<?x4x?xf32>, tensor<3xindex>
  %4 = shape.value_of %3 : tensor<?x4x?xf32>
  %5 = "test.concat"(%4, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
  %6 = shape.get_extent %0, %c0 : tensor<3xindex>, index -> index
  %7 = arith.addi %6, %c2 : index
  %8 = shape.from_extents %7, %c4, %1 : index, index, index
  %9 = shape.with_shape %5, %8 : tensor<?x4x?xf32>, !shape.shape
  %10 = shape.value_of %9 : tensor<?x4x?xf32>
  return %10 : tensor<?x4x?xf32>
}

// CHECK-LABEL:  func.func @two_dynamic_one_direct_shape
// CHECK-NEXT:     %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
// CHECK-NEXT:     %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
// CHECK-NEXT:     return %1 : tensor<?x4x?xf32>

// CHECK: shape.func private @shape_cal_1(%arg0: tensor<?x4x?xf32>) -> !shape.shape {
// CHECK-DAG:      %[[V0:.*]] = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
// CHECK-DAG:      %[[V1:.*]] = get_extent %[[V0]], %c2 : tensor<3xindex>, index -> index
// CHECK-DAG:      %[[V2:.*]] = get_extent %[[V0]], %c0 : tensor<3xindex>, index -> index
// CHECK-DAG:      %[[V3:.*]] = arith.addi %[[V2]], %c2 : index
// CHECK-DAG:      %[[V4:.*]] = from_extents %[[V3]], %c4, %[[V1]] : index, index, index
// CHECK-DAG:      return %[[V4]] : !shape.shape

// CHECK: shape.func private @shape_cal_0(%arg0: tensor<?x4x?xf32>) -> tensor<3xindex> {
// CHECK-DAG:   %0 = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
// CHECK-DAG:   return %0 : tensor<3xindex>

// -----

// Two dynamic shapes and they share the same shape.func
func.func @two_dynamic_share_same_shape(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) -> tensor<?x4x?xf32> {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %0 = shape.shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
  %1 = shape.get_extent %0, %c2 : tensor<3xindex>, index -> index
  %2 = "test.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
  %3 = shape.get_extent %0, %c0 : tensor<3xindex>, index -> index
  %4 = arith.addi %3, %c2 : index
  %5 = shape.from_extents %4, %c4, %1 : index, index, index
  %6 = shape.with_shape %2, %5 : tensor<?x4x?xf32>, !shape.shape
  %7 = shape.value_of %6 : tensor<?x4x?xf32>
  %8 = "test.abs"(%7) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
  %9 = shape.with_shape %8, %5 : tensor<?x4x?xf32>, !shape.shape
  %10 = shape.value_of %9 : tensor<?x4x?xf32>
  return %10 : tensor<?x4x?xf32>
}
// CHECK-LABEL: func.func @two_dynamic_share_same_shape
// CHECK-NEXT:     %0 = "test.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
// CHECK-NEXT:     %1 = "test.abs"(%0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
// CHECK-NEXT:     return %1 : tensor<?x4x?xf32>

// CHECK:       shape.func private @shape_cal_0(%arg0: tensor<?x4x?xf32>) -> !shape.shape {
// CHECK-DAG:     %[[V0:.*]] = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
// CHECK-DAG:     %[[V1:.*]] = get_extent %[[V0]], %c2 : tensor<3xindex>, index -> index
// CHECK-DAG:     %[[V2:.*]] = get_extent %[[V0]], %c0 : tensor<3xindex>, index -> index
// CHECK-DAG:     %[[V3:.*]] = arith.addi %[[V2]], %c2 : index
// CHECK-DAG:     %[[V4:.*]] = from_extents %[[V3]], %c4, %[[V1]] : index, index, index
// CHECK-DAG:     return %4 : !shape.shape
// CHECK-NOT: shape_cal_1

// -----

// There's an internal dynamic shape source, and two other dynamic shapes shares it
func.func @internal_dynamic_shape_source_shared(%arg0: tensor<?x4xf32>) -> tensor<?xi32> {
  %0 = "test.nonzero"(%arg0) : (tensor<?x4xf32>) -> tensor<?xi32>
  %1 = shape.shape_of %0 : tensor<?xi32> -> tensor<1xindex>
  %2 = shape.with_shape %0, %1 : tensor<?xi32>, tensor<1xindex>
  %3 = shape.value_of %2 : tensor<?xi32>
  %4 = "test.abs"(%3) : (tensor<?xi32>) -> tensor<?xi32>
  %5 = shape.with_shape %4, %1 : tensor<?xi32>, tensor<1xindex>
  %6 = shape.value_of %5 : tensor<?xi32>
  %7 = "test.negate"(%6) : (tensor<?xi32>) -> tensor<?xi32>
  %8 = shape.with_shape %7, %1 : tensor<?xi32>, tensor<1xindex>
  %9 = shape.value_of %8 : tensor<?xi32>
  return %9 : tensor<?xi32>
}
// CHECK-LABEL: func.func @internal_dynamic_shape_source_shared
// CHECK-NEXT:     %0 = "test.nonzero"(%arg0) : (tensor<?x4xf32>) -> tensor<?xi32>
// CHECK-NEXT:     %1 = "test.abs"(%0) : (tensor<?xi32>) -> tensor<?xi32>
// CHECK-NEXT:     %2 = "test.negate"(%1) : (tensor<?xi32>) -> tensor<?xi32>
// CHECK-NEXT:     return %2 : tensor<?xi32>

// CHECK:      shape.func private @shape_cal_0(%arg0: tensor<?xi32>) -> tensor<1xindex> {
// CHECK-NEXT:   %0 = shape_of %arg0 : tensor<?xi32> -> tensor<1xindex>
// CHECK-NEXT:   return %0 : tensor<1xindex>
// CHECK-NOT: shape_cal_1

// -----

// There's only a return op in the constructed shape.func
func.func @only_return_of_constructed_shape(%arg0: tensor<?x4xf32>, %arg1: tensor<1xindex>) -> tensor<?xi32> {
  %0 = "test.nonzero"(%arg0) : (tensor<?x4xf32>) -> tensor<?xi32>
  %1 = shape.with_shape %0, %arg1 : tensor<?xi32>, tensor<1xindex>
  %2 = shape.value_of %1 : tensor<?xi32>
  return %2 : tensor<?xi32>
}
// CHECK-LABEL: func.func @only_return_of_constructed_shape(%arg0: tensor<?x4xf32>, %arg1: tensor<1xindex>) -> tensor<?xi32> {
// CHECK-NEXT:   %0 = "test.nonzero"(%arg0) : (tensor<?x4xf32>) -> tensor<?xi32>
// CHECK-NEXT:   return %0 : tensor<?xi32>

// CHECK:      shape.func private @shape_cal_0(%arg0: tensor<1xindex>) -> tensor<1xindex> {
// CHECK-NEXT:   return %arg0 : tensor<1xindex>

// -----

// Shape computation part interleaves with general computation.
func.func @interleaved_shape_computation(%arg0: tensor<?x4x5xf32>, %arg1: tensor<?x4x5xf32>, %arg2: tensor<?x4x5xf32>) -> (tensor<?x4x5xf32>, index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %0 = shape.shape_of %arg0 : tensor<?x4x5xf32> -> tensor<3xindex>
  %1 = shape.shape_of %arg1 : tensor<?x4x5xf32> -> tensor<3xindex>
  %2 = shape.shape_of %arg2 : tensor<?x4x5xf32> -> tensor<3xindex>
  %3 = "test.concat"(%arg0, %arg1, %arg2) {axis = 0 : i64} : (tensor<?x4x5xf32>, tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
  %4 = shape.get_extent %0, %c0 : tensor<3xindex>, index -> index
  %5 = shape.get_extent %1, %c0 : tensor<3xindex>, index -> index
  %6 = shape.get_extent %2, %c0 : tensor<3xindex>, index -> index
  %7 = arith.addi %4, %5 : index
  %8 = arith.addi %7, %6 : index
  %9 = shape.from_extents %8, %c4, %c5 : index, index, index
  %10 = shape.with_shape %3, %9 : tensor<?x4x5xf32>, !shape.shape
  %11 = shape.value_of %10 : tensor<?x4x5xf32>
  return %11, %7 : tensor<?x4x5xf32>, index
}
// CHECK-LABEL: func.func @interleaved_shape_computation
// CHECK-DAG:   %[[V0:.*]] = shape.shape_of %arg0 : tensor<?x4x5xf32> -> tensor<3xindex>
// CHECK-DAG:   %[[V1:.*]] = shape.shape_of %arg1 : tensor<?x4x5xf32> -> tensor<3xindex>
// CHECK-DAG:   %[[V2:.*]] = "test.concat"(%arg0, %arg1, %arg2) {axis = 0 : i64} : (tensor<?x4x5xf32>, tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
// CHECK-DAG:   %[[V3:.*]] = shape.get_extent %[[V0]], %c0 : tensor<3xindex>, index -> index
// CHECK-DAG:   %[[V4:.*]] = shape.get_extent %[[V1]], %c0 : tensor<3xindex>, index -> index
// CHECK-DAG:   %[[V5:.*]] = arith.addi %[[V3]], %[[V4]] : index
// CHECK-DAG:   return %[[V2]], %[[V5]] : tensor<?x4x5xf32>, index

// CHECK:     shape.func private @shape_cal_0(%arg0: tensor<?x4x5xf32>, %arg1: index, %arg2: index) -> !shape.shape {
// CHECK-DAG:   %[[V0:.*]] = shape_of %arg0 : tensor<?x4x5xf32> -> tensor<3xindex>
// CHECK-DAG:   %[[V1:.*]] = get_extent %[[V0]], %arg1 : tensor<3xindex>, index -> index
// CHECK-DAG:   %[[V2:.*]] = arith.addi %arg2, %[[V1]] : index
// CHECK-DAG:   %[[V3:.*]] = from_extents %[[V2]], %c4, %c5 : index, index, index
// CHECK-DAG:   return %[[V3]] : !shape.shape

// -----

// There're multiple reused shape computations.
func.func @multiple_reused(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %0 = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
  %1 = shape.shape_of %arg1 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = "test.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %3 = "test.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %4 = shape.get_extent %0, %c0 : tensor<2xindex>, index -> index
  %5 = shape.get_extent %1, %c0 : tensor<2xindex>, index -> index
  %6 = arith.addi %4, %5 : index
  %7 = shape.from_extents %6, %c4 : index, index
  %8 = shape.with_shape %2, %7 : tensor<?x4xf32>, !shape.shape
  %9 = shape.with_shape %3, %7 : tensor<?x4xf32>, !shape.shape
  %10 = shape.value_of %8 : tensor<?x4xf32>
  %11 = shape.value_of %9 : tensor<?x4xf32>
  %12 = "test.concat"(%arg0, %2) {axis = 0 : i64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %13 = "test.concat"(%arg0, %3) {axis = 0 : i64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %14 = arith.addi %6, %4 : index
  %15 = shape.from_extents %14, %c4 : index, index
  %16 = shape.with_shape %12, %15 : tensor<?x4xf32>, !shape.shape
  %17 = shape.with_shape %13, %15 : tensor<?x4xf32>, !shape.shape
  %18 = shape.value_of %16 : tensor<?x4xf32>
  %19 = shape.value_of %17 : tensor<?x4xf32>
  return %10, %11, %18, %19 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}
// CHECK-LABEL: func.func @multiple_reused
// CHECK-DAG:     %[[V0:.*]] = "test.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:     %[[V1:.*]] = "test.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:     %[[V2:.*]] = "test.concat"(%arg0, %[[V0]]) {axis = 0 : i64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:     %[[V3:.*]] = "test.concat"(%arg0, %[[V1]]) {axis = 0 : i64} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:     return %[[V0]], %[[V1]], %[[V2]], %[[V3]] : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>

// CHECK:      shape.func private @shape_cal_1(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> !shape.shape {
// CHECK-DAG:    %[[V0:.*]] = shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK-DAG:    %[[V1:.*]] = shape_of %arg1 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK-DAG:    %[[V2:.*]] = get_extent %[[V0]], %c0 : tensor<2xindex>, index -> index
// CHECK-DAG:    %[[V3:.*]] = get_extent %[[V1]], %c0 : tensor<2xindex>, index -> index
// CHECK-DAG:    %[[V4:.*]] = arith.addi %[[V2]], %[[V3]] : index
// CHECK-DAG:    %[[V5:.*]] = arith.addi %[[V4]], %[[V2]] : index
// CHECK-DAG:    %[[V6:.*]] = from_extents %[[V5]], %c4 : index, index
// CHECK-DAG:    return %[[V6]] : !shape.shape

// CHECK:     shape.func private @shape_cal_0(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> !shape.shape {
// CHECK-DAG:   %[[V0:.*]] = shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK-DAG:   %[[V1:.*]] = shape_of %arg1 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK-DAG:   %[[V2:.*]] = get_extent %[[V0]], %c0 : tensor<2xindex>, index -> index
// CHECK-DAG:   %[[V3:.*]] = get_extent %[[V1]], %c0 : tensor<2xindex>, index -> index
// CHECK-DAG:   %[[V4:.*]] = arith.addi %[[V2]], %[[V3]] : index
// CHECK-DAG:   %[[V5:.*]] = from_extents %[[V4]], %c4 : index, index
// CHECK-DAG:   return %[[V5]] : !shape.shape

