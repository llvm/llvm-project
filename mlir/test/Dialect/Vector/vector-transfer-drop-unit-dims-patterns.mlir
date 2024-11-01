// RUN: mlir-opt %s --test-transform-dialect-interpreter | FileCheck %s

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

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.rank_reducing_subview_patterns
  } : !transform.op<"func.func">
}
