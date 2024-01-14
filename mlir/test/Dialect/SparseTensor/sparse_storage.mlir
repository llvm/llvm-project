// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification= | FileCheck %s

#SparseVector64 = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed),
  posWidth = 64,
  crdWidth = 64
}>

#SparseVector32 = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#trait_mul = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * b(i)"
}

// CHECK-LABEL: func @mul64(
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[P0:.*]] = memref.load %{{.*}}[%[[C0]]] : memref<?xi64>
// CHECK: %[[B0:.*]] = arith.index_cast %[[P0]] : i64 to index
// CHECK: %[[P1:.*]] = memref.load %{{.*}}[%[[C1]]] : memref<?xi64>
// CHECK: %[[B1:.*]] = arith.index_cast %[[P1]] : i64 to index
// CHECK: scf.for %[[I:.*]] = %[[B0]] to %[[B1]] step %[[C1]] {
// CHECK:   %[[IND0:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xi64>
// CHECK:   %[[INDC:.*]] = arith.index_cast %[[IND0]] : i64 to index
// CHECK:   %[[VAL0:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xf64>
// CHECK:   %[[VAL1:.*]] = memref.load %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK:   %[[MUL:.*]] = arith.mulf %[[VAL0]], %[[VAL1]] : f64
// CHECK:   store %[[MUL]], %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK: }
func.func @mul64(%arga: tensor<32xf64, #SparseVector64>, %argb: tensor<32xf64>, %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait_mul
     ins(%arga, %argb: tensor<32xf64, #SparseVector64>, tensor<32xf64>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL: func @mul32(
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[P0:.*]] = memref.load %{{.*}}[%[[C0]]] : memref<?xi32>
// CHECK: %[[Z0:.*]] = arith.extui %[[P0]] : i32 to i64
// CHECK: %[[B0:.*]] = arith.index_cast %[[Z0]] : i64 to index
// CHECK: %[[P1:.*]] = memref.load %{{.*}}[%[[C1]]] : memref<?xi32>
// CHECK: %[[Z1:.*]] = arith.extui %[[P1]] : i32 to i64
// CHECK: %[[B1:.*]] = arith.index_cast %[[Z1]] : i64 to index
// CHECK: scf.for %[[I:.*]] = %[[B0]] to %[[B1]] step %[[C1]] {
// CHECK:   %[[IND0:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xi32>
// CHECK:   %[[ZEXT:.*]] = arith.extui %[[IND0]] : i32 to i64
// CHECK:   %[[INDC:.*]] = arith.index_cast %[[ZEXT]] : i64 to index
// CHECK:   %[[VAL0:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xf64>
// CHECK:   %[[VAL1:.*]] = memref.load %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK:   %[[MUL:.*]] = arith.mulf %[[VAL0]], %[[VAL1]] : f64
// CHECK:   store %[[MUL]], %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK: }
func.func @mul32(%arga: tensor<32xf64, #SparseVector32>, %argb: tensor<32xf64>, %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait_mul
     ins(%arga, %argb: tensor<32xf64, #SparseVector32>, tensor<32xf64>)
    outs(%argx: tensor<32xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}
