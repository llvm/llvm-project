// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

func.func @transfer_read_rank_reducing(
      %arg : memref<1x1x3x2xi8, strided<[6, 6, 2, 1], offset: ?>>) -> vector<3x2xi8> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0 : i8
    %v = vector.transfer_read %arg[%c0, %c0, %c0, %c0], %cst :
      memref<1x1x3x2xi8, strided<[6, 6, 2, 1], offset: ?>>, vector<3x2xi8>
    return %v : vector<3x2xi8>
}
// CHECK-LABEL: func @transfer_read_rank_reducing
//  CHECK-SAME:     %[[ARG:.+]]: memref<1x1x3x2xi8
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0, 0, 0] [1, 1, 3, 2] [1, 1, 1, 1]
//  CHECK-SAME:     memref<1x1x3x2xi8, {{.*}}> to memref<3x2xi8, {{.*}}>
//       CHECK:   vector.transfer_read %[[SUBVIEW]]

func.func @transfer_write_rank_reducing(%arg : memref<1x1x3x2xi8, strided<[6, 6, 2, 1], offset: ?>>, %vec : vector<3x2xi8>) {
    %c0 = arith.constant 0 : index
    vector.transfer_write %vec, %arg [%c0, %c0, %c0, %c0] :
      vector<3x2xi8>, memref<1x1x3x2xi8, strided<[6, 6, 2, 1], offset: ?>>
    return
}
// CHECK-LABEL: func @transfer_write_rank_reducing
//  CHECK-SAME:     %[[ARG:.+]]: memref<1x1x3x2xi8
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0, 0, 0] [1, 1, 3, 2] [1, 1, 1, 1]
//  CHECK-SAME:     memref<1x1x3x2xi8, {{.*}}> to memref<3x2xi8, {{.*}}>
//       CHECK:   vector.transfer_write %{{.*}}, %[[SUBVIEW]]

func.func @transfer_read_and_vector_rank_reducing(
      %arg : memref<1x1x3x2x1xf32>) -> vector<3x2x1xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %v = vector.transfer_read %arg[%c0, %c0, %c0, %c0, %c0], %cst :
      memref<1x1x3x2x1xf32>, vector<3x2x1xf32>
    return %v : vector<3x2x1xf32>
}
// CHECK-LABEL: func @transfer_read_and_vector_rank_reducing
//  CHECK-SAME:     %[[ARG:.+]]: memref<1x1x3x2x1xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0, 0, 0, 0] [1, 1, 3, 2, 1] [1, 1, 1, 1, 1]
//  CHECK-SAME:     memref<1x1x3x2x1xf32> to memref<3x2xf32>
//       CHECK:   vector.transfer_read %[[SUBVIEW]]{{.*}} {in_bounds = [true, true]} : memref<3x2xf32>, vector<3x2xf32>

func.func @transfer_write_and_vector_rank_reducing(
      %arg : memref<1x1x3x2x1xf32>,
      %vec : vector<3x2x1xf32>) {
    %c0 = arith.constant 0 : index
    vector.transfer_write %vec, %arg [%c0, %c0, %c0, %c0, %c0] :
      vector<3x2x1xf32>, memref<1x1x3x2x1xf32>
    return
}
// CHECK-LABEL: func @transfer_write_and_vector_rank_reducing
//  CHECK-SAME:     %[[ARG:.+]]: memref<1x1x3x2x1xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0, 0, 0, 0] [1, 1, 3, 2, 1] [1, 1, 1, 1, 1]
//  CHECK-SAME:     memref<1x1x3x2x1xf32> to memref<3x2xf32>
//       CHECK:   vector.transfer_write %{{.*}}, %[[SUBVIEW]]{{.*}} {in_bounds = [true, true]} : vector<3x2xf32>, memref<3x2xf32>

func.func @transfer_read_and_vector_rank_reducing_to_0d(
      %arg : memref<1x1x1x1x1xf32>) -> vector<1x1x1xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %v = vector.transfer_read %arg[%c0, %c0, %c0, %c0, %c0], %cst :
      memref<1x1x1x1x1xf32>, vector<1x1x1xf32>
    return %v : vector<1x1x1xf32>
}
// CHECK-LABEL: func @transfer_read_and_vector_rank_reducing_to_0d
//  CHECK-SAME:     %[[MEMREF:.+]]: memref<1x1x1x1x1xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[MEMREF]][0, 0, 0, 0, 0] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1] : memref<1x1x1x1x1xf32> to memref<f32>
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[SUBVIEW]]{{.*}} : memref<f32>, vector<f32>
//       CHECK:   vector.shape_cast %[[READ]] : vector<f32> to vector<1x1x1xf32>

func.func @transfer_write_and_vector_rank_reducing_to_0d(
      %arg : memref<1x1x1x1x1xf32>,
      %vec : vector<1x1x1xf32>) {
    %c0 = arith.constant 0 : index
    vector.transfer_write %vec, %arg [%c0, %c0, %c0, %c0, %c0] :
      vector<1x1x1xf32>, memref<1x1x1x1x1xf32>
    return
}
// CHECK-LABEL: func @transfer_write_and_vector_rank_reducing_to_0d
//  CHECK-SAME:     %[[MEMREF:.+]]: memref<1x1x1x1x1xf32>, %[[VECTOR:.+]]: vector<1x1x1xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[MEMREF]][0, 0, 0, 0, 0] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1] : memref<1x1x1x1x1xf32> to memref<f32>
//       CHECK:   %[[SHCAST:.+]] = vector.shape_cast %[[VECTOR]] : vector<1x1x1xf32> to vector<f32>
//       CHECK:   vector.transfer_write %[[SHCAST]], %[[SUBVIEW]]{{.*}} : vector<f32>, memref<f32>

func.func @transfer_read_dynamic_rank_reducing(
      %arg : memref<?x1xi8, strided<[?, ?], offset: ?>>) -> vector<[16]x1xi8> {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i8
    %v = vector.transfer_read %arg[%c0, %c0], %pad {in_bounds = [true, true]} :
      memref<?x1xi8, strided<[?, ?], offset: ?>>, vector<[16]x1xi8>
    return %v : vector<[16]x1xi8>
}
// CHECK-LABEL: func @transfer_read_dynamic_rank_reducing
//  CHECK-SAME:     %[[ARG:.+]]: memref<?x1xi8
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[DIM0:.+]] = memref.dim %[[ARG]], %[[C0]] : memref<?x1xi8, strided<[?, ?], offset: ?>>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0] [%[[DIM0]], 1] [1, 1] : memref<?x1xi8, {{.*}}> to memref<?xi8, {{.*}}>
//       CHECK:   vector.transfer_read %[[SUBVIEW]]{{.*}} : memref<?xi8, {{.*}}>, vector<[16]xi8>

func.func @masked_transfer_read_dynamic_rank_reducing_1(
      %arg : memref<?x1xi8, strided<[?, ?], offset: ?>>,
      %mask_dim0 : index) -> vector<[16]x1xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %pad = arith.constant 0 : i8
    %mask = vector.create_mask %mask_dim0, %c1 : vector<[16]x1xi1>
    %v = vector.transfer_read %arg[%c0, %c0], %pad, %mask {in_bounds = [true, true]} :
      memref<?x1xi8, strided<[?, ?], offset: ?>>, vector<[16]x1xi8>
    return %v : vector<[16]x1xi8>
}
// CHECK-LABEL: func @masked_transfer_read_dynamic_rank_reducing_1
//  CHECK-SAME:     %[[ARG:.+]]: memref<?x1xi8
//  CHECK-SAME:     %[[MASK_DIM0:.+]]: index
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[PAD:.+]] = arith.constant 0 : i8
//       CHECK:   %[[MASK:.+]] = vector.create_mask %[[MASK_DIM0]] : vector<[16]xi1>
//       CHECK:   %[[DIM0:.+]] = memref.dim %[[ARG]], %[[C0]] : memref<?x1xi8, strided<[?, ?], offset: ?>>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0] [%[[DIM0]], 1] [1, 1] : memref<?x1xi8, {{.*}}> to memref<?xi8, {{.*}}>
//       CHECK:   vector.transfer_read %[[SUBVIEW]][{{.*}}], %[[PAD]], %[[MASK]] {in_bounds = [true]} : memref<?xi8, {{.*}}>, vector<[16]xi8>

func.func @masked_transfer_read_dynamic_rank_reducing_2(
      %arg : memref<1x?x3x1x?x1xi8, strided<[?, ?, ?, ?, ?, ?], offset: ?>>,
      %mask_dim1 : index, %mask_dim4 : index) -> vector<1x[1]x3x1x[16]x1xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %pad = arith.constant 0 : i8
    %mask = vector.create_mask %c1, %mask_dim1, %c2, %c1, %mask_dim4, %c1 : vector<1x[1]x3x1x[16]x1xi1>
    %v = vector.transfer_read %arg[%c0, %c0, %c0, %c0, %c0, %c0], %pad, %mask {in_bounds = [true, true, true, true, true, true]} :
      memref<1x?x3x1x?x1xi8, strided<[?, ?, ?, ?, ?, ?], offset: ?>>, vector<1x[1]x3x1x[16]x1xi8>
    return %v : vector<1x[1]x3x1x[16]x1xi8>
}
// CHECK-LABEL: func @masked_transfer_read_dynamic_rank_reducing_2
//  CHECK-SAME:     %[[ARG:.+]]: memref<1x?x3x1x?x1xi8
//  CHECK-SAME:     %[[MASK_DIM1:.+]]: index, %[[MASK_DIM4:.+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[PAD:.+]] = arith.constant 0 : i8
//       CHECK:   %[[MASK:.+]] = vector.create_mask %[[MASK_DIM1]], %[[C2]], %[[MASK_DIM4]] : vector<[1]x3x[16]xi1>
//       CHECK:   %[[DIM1:.+]] = memref.dim %[[ARG]], %[[C1]] : memref<1x?x3x1x?x1xi8, strided<[?, ?, ?, ?, ?, ?], offset: ?>>
//       CHECK:   %[[DIM4:.+]] = memref.dim %[[ARG]], %[[C4]] : memref<1x?x3x1x?x1xi8, strided<[?, ?, ?, ?, ?, ?], offset: ?>>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0, 0, 0, 0, 0] [1, %[[DIM1]], 3, 1, %[[DIM4]], 1] [1, 1, 1, 1, 1, 1] : memref<1x?x3x1x?x1xi8, {{.*}}> to memref<?x3x?xi8, {{.*}}>
//       CHECK:   vector.transfer_read %[[SUBVIEW]][{{.*}}], %[[PAD]], %[[MASK]] {in_bounds = [true, true, true]} : memref<?x3x?xi8, {{.*}}>, vector<[1]x3x[16]xi8>

func.func @masked_transfer_write_and_vector_rank_reducing(
      %arg : memref<1x1x3x1x16x1xf32>,
      %vec : vector<1x3x1x16x1xf32>,
      %mask_dim1 : index,
      %mask_dim2 : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mask = vector.create_mask %c1, %mask_dim1, %c1, %mask_dim2, %c1 : vector<1x3x1x16x1xi1>
    vector.transfer_write %vec, %arg[%c0, %c0, %c0, %c0, %c0, %c0], %mask :
      vector<1x3x1x16x1xf32>, memref<1x1x3x1x16x1xf32>
    return
}
// CHECK-LABEL: func @masked_transfer_write_and_vector_rank_reducing
//  CHECK-SAME:     %[[ARG:.+]]: memref<1x1x3x1x16x1xf32>
//  CHECK-SAME:     {{.*}}: vector<1x3x1x16x1xf32>,
//  CHECK-SAME:     %[[MASKDIM1:.+]]: index,
//  CHECK-SAME:     %[[MASKDIM2:.+]]: index
//       CHECK:   %[[MASK:.+]] = vector.create_mask %[[MASKDIM1]], %[[MASKDIM2]] : vector<3x16xi1>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0, 0, 0, 0, 0] [1, 1, 3, 1, 16, 1] [1, 1, 1, 1, 1, 1]
//  CHECK-SAME:     memref<1x1x3x1x16x1xf32> to memref<3x16xf32>
//       CHECK:   vector.transfer_write %{{.*}}, %[[SUBVIEW]]{{.*}}, %[[MASK]] {in_bounds = [true, true]} : vector<3x16xf32>, memref<3x16xf32>

func.func @masked_transfer_write_dynamic_rank_reducing(
      %arg : memref<?x1xi8, strided<[?, ?], offset: ?>>,
      %vec : vector<[16]x1xi8>,
      %mask_dim0 : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %pad = arith.constant 0 : i8
    %mask = vector.create_mask %mask_dim0, %c1 : vector<[16]x1xi1>
    vector.transfer_write %vec, %arg[%c0, %c0], %mask {in_bounds = [true, true]} :
      vector<[16]x1xi8>, memref<?x1xi8, strided<[?, ?], offset: ?>>
    return
}
// CHECK-LABEL: func @masked_transfer_write_dynamic_rank_reducing
//  CHECK-SAME:     %[[ARG:.+]]: memref<?x1xi8
//  CHECK-SAME:     %{{.*}}: vector<[16]x1xi8>,
//  CHECK-SAME:     %[[MASK_DIM0:.+]]: index
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[MASK:.+]] = vector.create_mask %[[MASK_DIM0]] : vector<[16]xi1>
//       CHECK:   %[[DIM0:.+]] = memref.dim %[[ARG]], %[[C0]] : memref<?x1xi8, strided<[?, ?], offset: ?>>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0] [%[[DIM0]], 1] [1, 1] : memref<?x1xi8, {{.*}}> to memref<?xi8, {{.*}}>
//       CHECK:   vector.transfer_write {{.*}}, %[[SUBVIEW]][%[[C0]]], %[[MASK]] {in_bounds = [true]} : vector<[16]xi8>, memref<?xi8, {{.*}}>

/// Only masks operands of vector.create_mask are currently supported.
func.func @unsupported_masked_transfer_read_dynamic_rank_reducing_1(
      %arg : memref<?x1xi8, strided<[?, ?], offset: ?>>,
      %mask : vector<[16]x1xi1>) -> vector<[16]x1xi8> {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i8
    %v = vector.transfer_read %arg[%c0, %c0], %pad, %mask {in_bounds = [true, true]} :
      memref<?x1xi8, strided<[?, ?], offset: ?>>, vector<[16]x1xi8>
    return %v : vector<[16]x1xi8>
}
// CHECK-LABEL: func @unsupported_masked_transfer_read_dynamic_rank_reducing_1
//  CHECK-SAME:     %[[ARG:.+]]: memref<?x1xi8
//   CHECK-NOT: vector.create_mask
//   CHECK-NOT: memref.subview
//       CHECK: vector.transfer_read %[[ARG]]

/// Unit dim mask must be constant of 1.
func.func @unsupported_masked_transfer_read_dynamic_rank_reducing_2(
      %arg : memref<?x1xi8, strided<[?, ?], offset: ?>>,
      %mask_dim0 : index, %mask_dim1 : index) -> vector<[16]x1xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %pad = arith.constant 0 : i8
    %mask = vector.create_mask %mask_dim0, %mask_dim1 : vector<[16]x1xi1>
    %v = vector.transfer_read %arg[%c0, %c0], %pad, %mask {in_bounds = [true, true]} :
      memref<?x1xi8, strided<[?, ?], offset: ?>>, vector<[16]x1xi8>
    return %v : vector<[16]x1xi8>
}
// CHECK-LABEL: func @unsupported_masked_transfer_read_dynamic_rank_reducing_2
//  CHECK-SAME:     %[[ARG:.+]]: memref<?x1xi8
//   CHECK-NOT: memref.subview
//       CHECK: vector.transfer_read {{.*}} vector<[16]x1xi8>

/// Unit dim must be non-scalable.
func.func @masked_transfer_read_dynamic_rank_reducing_scalable_unit_dim(
      %arg : memref<?x1xi8, strided<[?, ?], offset: ?>>,
      %mask_dim0 : index) -> vector<[16]x[1]xi8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %pad = arith.constant 0 : i8
    %mask = vector.create_mask %mask_dim0, %c1 : vector<[16]x[1]xi1>
    %v = vector.transfer_read %arg[%c0, %c0], %pad, %mask {in_bounds = [true, true]} :
      memref<?x1xi8, strided<[?, ?], offset: ?>>, vector<[16]x[1]xi8>
    return %v : vector<[16]x[1]xi8>
}
// CHECK-LABEL: func @masked_transfer_read_dynamic_rank_reducing_scalable_unit_dim
//  CHECK-SAME:     %[[ARG:.+]]: memref<?x1xi8
//   CHECK-NOT: memref.subview
//       CHECK: vector.transfer_read {{.*}} vector<[16]x[1]xi8>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%func_op: !transform.op<"func.func"> {transform.readonly}) {
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.rank_reducing_subview_patterns
    } : !transform.op<"func.func">
    transform.yield
  }
}
