// RUN: mlir-opt %s -convert-linalg-to-std | FileCheck %s

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d2 * s2 + d1)>
// CHECK-DAG: #[[$map4:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 * s1 + s0 + d1 * s2 + d0)>
// CHECK-DAG: #[[$map6:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
// CHECK-DAG: #[[$map7:.*]] = affine_map<()[s0] -> (s0)>
// CHECK-DAG: #[[$map8:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>

func @dot(%arg0: memref<?xf32, offset: ?, strides: [1]>,
          %arg1: memref<?xf32, offset: ?, strides: [1]>,
          %arg2: memref<f32>) {
  linalg.dot ins(%arg0, %arg1: memref<?xf32, offset: ?, strides: [1]>,
                               memref<?xf32, offset: ?, strides: [1]>)
             outs(%arg2: memref<f32>)
  return
}
// CHECK-LABEL: func @dot(
//  CHECK-SAME: %[[arg0:[a-zA-z0-9]*]]: memref<?xf32, #[[$map0]]>,
//  CHECK-SAME: %[[arg1:[a-zA-z0-9]*]]: memref<?xf32, #[[$map0]]>,
//  CHECK-SAME: %[[arg2:[a-zA-z0-9]*]]: memref<f32>) {
//       CHECK:   %[[o0:.*]] = memref.cast %[[arg0]] :
//  CHECK-SAME:     memref<?xf32, #[[$map0]]> to memref<?xf32, #[[$map6]]>
//       CHECK:   %[[o1:.*]] = memref.cast %[[arg1]] :
//  CHECK-SAME:     memref<?xf32, #[[$map0]]> to memref<?xf32, #[[$map6]]>
//       CHECK:   %[[o2:.*]] = memref.cast %[[arg2]] :
//  CHECK-SAME:     memref<f32> to memref<f32, #[[$map7]]>
//       CHECK:   call @linalg_dot_viewsxf32_viewsxf32_viewf32(
//  CHECK-SAME:     %[[o0]], %[[o1]], %[[o2]]) :
//  CHECK-SAME:   memref<?xf32, #[[$map6]]>, memref<?xf32, #[[$map6]]>, memref<f32, #[[$map7]]>

func @copy(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy(
//  CHECK-SAME: %[[arg0:[a-zA-z0-9]*]]: memref<?x?x?xf32, #[[$map1]]>,
//  CHECK-SAME: %[[arg1:[a-zA-z0-9]*]]: memref<?x?x?xf32, #[[$map1]]>) {
//       CHECK:   %[[o0:.*]] = memref.cast %[[arg0]] :
//  CHECK-SAME:     memref<?x?x?xf32, #[[$map1]]> to memref<?x?x?xf32, #[[$map8]]>
//       CHECK:   %[[o1:.*]] = memref.cast %[[arg1]] :
//  CHECK-SAME:     memref<?x?x?xf32, #[[$map1]]> to memref<?x?x?xf32, #[[$map8]]>
//       CHECK:   call @linalg_copy_viewsxsxsxf32_viewsxsxsxf32(%[[o0]], %[[o1]]) :
//  CHECK-SAME:   memref<?x?x?xf32, #[[$map8]]>, memref<?x?x?xf32, #[[$map8]]>

func @copy_transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = affine_map<(i, j, k) -> (i, k, j)>,
                             outputPermutation = affine_map<(i, j, k) -> (k, j, i)>}
    : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy_transpose(
//  CHECK-SAME: %[[arg0:[a-zA-z0-9]*]]: memref<?x?x?xf32, #[[$map1]]>,
//  CHECK-SAME: %[[arg1:[a-zA-z0-9]*]]: memref<?x?x?xf32, #[[$map1]]>) {
//       CHECK:   %[[t0:.*]] = memref.transpose %[[arg0]]
//  CHECK-SAME:     (d0, d1, d2) -> (d0, d2, d1) : memref<?x?x?xf32, #[[$map1]]>
//       CHECK:   %[[t1:.*]] = memref.transpose %[[arg1]]
//  CHECK-SAME:     (d0, d1, d2) -> (d2, d1, d0) : memref<?x?x?xf32, #[[$map1]]>
//       CHECK:   %[[o0:.*]] = memref.cast %[[t0]] :
//  CHECK-SAME:     memref<?x?x?xf32, #[[$map2]]> to memref<?x?x?xf32, #[[$map8]]>
//       CHECK:   %[[o1:.*]] = memref.cast %[[t1]] :
//  CHECK-SAME:     memref<?x?x?xf32, #[[$map4]]> to memref<?x?x?xf32, #[[$map8]]>
//       CHECK:   call @linalg_copy_viewsxsxsxf32_viewsxsxsxf32(%[[o0]], %[[o1]]) :
//  CHECK-SAME:   memref<?x?x?xf32, #[[$map8]]>, memref<?x?x?xf32, #[[$map8]]>

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmul_trait = {
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = #matmul_accesses,
  library_call = "external_outerproduct_matmul"
}

!vector_type_A = type vector<4xf32>
!vector_type_B = type vector<4xf32>
!vector_type_C = type vector<4x4xf32>

!matrix_type_A = type memref<?x?x!vector_type_A>
!matrix_type_B = type memref<?x?x!vector_type_B>
!matrix_type_C = type memref<?x?x!vector_type_C>

func @matmul_vec_impl(%A: !matrix_type_A, %B: !matrix_type_B, %C: !matrix_type_C) {
  linalg.generic #matmul_trait
      ins(%A, %B : !matrix_type_A, !matrix_type_B)
     outs(%C : !matrix_type_C) {
    ^bb0(%a: !vector_type_A, %b: !vector_type_B, %c: !vector_type_C):
      %d = vector.outerproduct %a, %b, %c: !vector_type_A, !vector_type_B
      linalg.yield %d: !vector_type_C
  }
  return
}
// CHECK-LABEL: func @matmul_vec_impl(
// CHECK:  call @external_outerproduct_matmul(%{{.*}}) :
