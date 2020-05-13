// RUN: mlir-opt %s -convert-linalg-to-llvm | FileCheck %s
// RUN: mlir-opt %s -convert-linalg-to-loops | FileCheck %s --check-prefix=LLVM-LOOPS

func @range(%arg0: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%arg0:%c1 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%{{.*}}: !llvm.i64) {
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:   llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:   llvm.mlir.undef : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">

func @slice(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: !linalg.range) {
  %1 = linalg.slice %arg0[%arg1] : memref<?xf32, offset: ?, strides: [1]>, !linalg.range, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @slice
//   insert data ptr for slice op
//       CHECK:   llvm.extractvalue %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.add %{{.*}}, %{{.*}} : !llvm.i64
//    insert offset
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.mlir.constant(0 : index)
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ i64, i64, i64 }">
//    get size[0] from parent view
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//    compute size[0] bounded by parent view's size[0]
//  CHECK-NEXT:   llvm.sub %{{.*}}, %{{.*}} : !llvm.i64
//    bound below by 0
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i64
//    compute stride[0] using bounded size
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : !llvm.i64
//    insert size and stride
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">

func @dot(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// CHECK-LABEL: func @dot
// CHECK:   llvm.call @linalg_dot_viewsxf32_viewsxf32_viewf32(%{{.*}}) :
// CHECK-SAME: !llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64
// CHECK-SAME: !llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64
// CHECK-SAME: !llvm<"float*">, !llvm<"float*">, !llvm.i64

func @slice_with_range_and_index(%arg0: memref<?x?xf64, offset: ?, strides: [?, 1]>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%c1:%c1 : !linalg.range
  loop.for %i0 = %c0 to %c1 step %c1 {
    %1 = linalg.slice %arg0[%i0, %R] : memref<?x?xf64, offset: ?, strides: [?, 1]>, index, !linalg.range, memref<?xf64, offset: ?, strides: [1]>
  }
  return
}
// CHECK-LABEL: func @slice_with_range_and_index
// loop-body.
//       CHECK:   llvm.mlir.undef : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[4, 0] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[4, 1] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ double*, double*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}[2] : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i64, i64 }">
//       CHECK:   llvm.insertvalue %{{.*}}[3, 0] : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}[4, 0] : !llvm<"{ double*, double*, i64, [1 x i64], [1 x i64] }">

func @copy(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy
//       CHECK:   llvm.call @linalg_copy_viewsxsxsxf32_viewsxsxsxf32({{.*}}) :
//  CHECK-SAME:   !llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64
//  CHECK-SAME:   !llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64

func @transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  %0 = linalg.transpose %arg0 (i, j, k) -> (k, i, j) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @transpose
//       CHECK:   llvm.mlir.undef : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">

func @copy_transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = affine_map<(i, j, k) -> (i, k, j)>,
                             outputPermutation = affine_map<(i, j, k) -> (k, j, i)>}
    : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy
// Transpose input
//         CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
// Transpose output
//         CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//         CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
// Call external copy.
//         CHECK:   llvm.call @linalg_copy_viewsxsxsxf32_viewsxsxsxf32

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmul_trait = {
  args_in = 2,
  args_out = 1,
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
  linalg.generic #matmul_trait %A, %B, %C {
    ^bb0(%a: !vector_type_A, %b: !vector_type_B, %c: !vector_type_C):
      %d = vector.outerproduct %a, %b, %c: !vector_type_A, !vector_type_B
      linalg.yield %d: !vector_type_C
  } : !matrix_type_A, !matrix_type_B, !matrix_type_C

  return
}
// CHECK-LABEL: func @matmul_vec_impl(
// CHECK:  llvm.call @external_outerproduct_matmul(%{{.*}}) :
// CHECK-SAME: !llvm<"<4 x float>*">, !llvm<"<4 x float>*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64
// CHECK-SAME: !llvm<"<4 x float>*">, !llvm<"<4 x float>*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64
// CHECK-SAME: !llvm<"[4 x <4 x float>]*">, !llvm<"[4 x <4 x float>]*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64

// LLVM-LOOPS-LABEL: func @matmul_vec_impl(
//  LLVM-LOOPS-SAME: %[[A:.*0]]: memref<?x?xvector<4xf32>>,
//  LLVM-LOOPS-SAME: %[[B:.*1]]: memref<?x?xvector<4xf32>>,
//  LLVM-LOOPS-SAME: %[[C:.*2]]: memref<?x?xvector<4x4xf32>>)
// LLVM-LOOPS: %[[C0:.*]] = constant 0 : index
// LLVM-LOOPS: %[[C1:.*]] = constant 1 : index
// LLVM-LOOPS: %[[T0:.*]] = dim %[[A]], 0 : memref<?x?xvector<4xf32>>
// LLVM-LOOPS: %[[T1:.*]] = dim %[[A]], 1 : memref<?x?xvector<4xf32>>
// LLVM-LOOPS: %[[T2:.*]] = dim %[[B]], 1 : memref<?x?xvector<4xf32>>
// LLVM-LOOPS: loop.for %[[I:.*]] = %[[C0]] to %[[T0]] step %[[C1]] {
// LLVM-LOOPS: loop.for %[[J:.*]] = %[[C0]] to %[[T2]] step %[[C1]] {
// LLVM-LOOPS: loop.for %[[K:.*]] = %[[C0]] to %[[T1]] step %[[C1]] {
// LLVM-LOOPS:   %[[T3:.*]] = load %[[A]][%[[I]], %[[K]]] : memref<?x?xvector<4xf32>>
// LLVM-LOOPS:   %[[T4:.*]] = load %[[B]][%[[K]], %[[J]]] : memref<?x?xvector<4xf32>>
// LLVM-LOOPS:   %[[T5:.*]] = load %[[C]][%[[I]], %[[J]]] : memref<?x?xvector<4x4xf32>>
// LLVM-LOOPS:   %[[T6:.*]] = vector.outerproduct %3, %4, %5 : vector<4xf32>, vector<4xf32>
// LLVM-LOOPS:   store %[[T6]], %[[C]][%[[I]], %[[J]]] : memref<?x?xvector<4x4xf32>>

#indexed_matmul_trait = {
  args_in = 2,
  args_out = 1,
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = #matmul_accesses,
  library_call = "external_indexed_outerproduct_matmul"
}
func @matmul_vec_indexed(%A: !matrix_type_A,
                         %B: !matrix_type_B,
                         %C: !matrix_type_C) {
  linalg.indexed_generic #indexed_matmul_trait %A, %B, %C {
    ^bb0(%i: index, %j: index, %k: index,
         %a: !vector_type_A, %b: !vector_type_B, %c: !vector_type_C):
      %d = vector.outerproduct %a, %b, %c: !vector_type_A, !vector_type_B
      linalg.yield %d: !vector_type_C
  } : !matrix_type_A, !matrix_type_B, !matrix_type_C
  return
}
// CHECK-LABEL: func @matmul_vec_indexed(
//   CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//   CHECK: llvm.call @external_indexed_outerproduct_matmul(%[[ZERO]], %[[ZERO]], %[[ZERO]], %{{.*}}) :
// CHECK-SAME: !llvm<"<4 x float>*">, !llvm<"<4 x float>*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64
// CHECK-SAME: !llvm<"<4 x float>*">, !llvm<"<4 x float>*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64
// CHECK-SAME: !llvm<"[4 x <4 x float>]*">, !llvm<"[4 x <4 x float>]*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64

func @reshape_static_expand(%arg0: memref<3x4x5xf32>) -> memref<1x3x4x1x5xf32> {
  // Reshapes that expand a contiguous tensor with some 1's.
  %0 = linalg.reshape %arg0 [affine_map<(i, j, k, l, m) -> (i, j)>,
                             affine_map<(i, j, k, l, m) -> (k)>,
                             affine_map<(i, j, k, l, m) -> (l, m)>] :
    memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
  return %0 : memref<1x3x4x1x5xf32>
}
// CHECK-LABEL: func @reshape_static_expand
//       CHECK:    llvm.mlir.undef : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.extractvalue %{{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.extractvalue %{{.*}}[2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(3 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(4 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 3] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(5 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 4] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(60 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(20 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(5 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 2] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(5 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 3] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 4] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">

func @reshape_static_collapse(%arg0: memref<1x3x4x1x5xf32>) -> memref<3x4x5xf32> {
  %0 = linalg.reshape %arg0 [affine_map<(i, j, k, l, m) -> (i, j)>,
                             affine_map<(i, j, k, l, m) -> (k)>,
                             affine_map<(i, j, k, l, m) -> (l, m)>] :
    memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
  return %0 : memref<3x4x5xf32>
}
// CHECK-LABEL: func @reshape_static_collapse
//       CHECK:    llvm.mlir.undef : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.extractvalue %{{.*}}[1] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.extractvalue %{{.*}}[2] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.mlir.constant(3 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.mlir.constant(4 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.mlir.constant(5 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.mlir.constant(20 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.mlir.constant(5 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:    llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">

func @reshape_fold_zero_dim(%arg0 : memref<1x1xf32>) -> memref<f32> {
  %0 = linalg.reshape %arg0 [] : memref<1x1xf32> into memref<f32>
  return %0 : memref<f32>
}
// CHECK-LABEL: func @reshape_fold_zero_dim
//       CHECK:   llvm.mlir.undef : !llvm<"{ float*, float*, i64 }">
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, float*, i64 }">
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, float*, i64 }">
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ float*, float*, i64 }">

func @reshape_expand_zero_dim(%arg0 : memref<f32>) -> memref<1x1xf32> {
  %0 = linalg.reshape %arg0 [] : memref<f32> into memref<1x1xf32>
  return %0 : memref<1x1xf32>
}
// CHECK-LABEL: func @reshape_expand_zero_dim
//       CHECK:   llvm.mlir.undef : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, float*, i64 }">
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm<"{ float*, float*, i64 }">
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm<"{ float*, float*, i64 }">
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK:   llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
