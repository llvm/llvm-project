// RUN: mlir-opt %s -test-sparsification="ptr-type=1 ind-type=1" | \
// RUN:   FileCheck %s --check-prefix=CHECK-TYPE0
// RUN: mlir-opt %s -test-sparsification="ptr-type=1 ind-type=2" | \
// RUN:   FileCheck %s --check-prefix=CHECK-TYPE1
// RUN: mlir-opt %s -test-sparsification="ptr-type=2 ind-type=1" | \
// RUN:   FileCheck %s --check-prefix=CHECK-TYPE2
// RUN: mlir-opt %s -test-sparsification="ptr-type=2 ind-type=2" | \
// RUN:   FileCheck %s --check-prefix=CHECK-TYPE3

#trait_mul_1d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> (i)>   // x (out)
  ],
  sparse = [
    [ "S" ],  // a
    [ "D" ],  // b
    [ "D" ]   // x
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * b(i)"
}

// CHECK-TYPE0-LABEL: func @mul_dd(
// CHECK-TYPE0: %[[C0:.*]] = constant 0 : index
// CHECK-TYPE0: %[[C1:.*]] = constant 1 : index
// CHECK-TYPE0: %[[P0:.*]] = load %{{.*}}[%[[C0]]] : memref<?xi64>
// CHECK-TYPE0: %[[B0:.*]] = index_cast %[[P0]] : i64 to index
// CHECK-TYPE0: %[[P1:.*]] = load %{{.*}}[%[[C1]]] : memref<?xi64>
// CHECK-TYPE0: %[[B1:.*]] = index_cast %[[P1]] : i64 to index
// CHECK-TYPE0: scf.for %[[I:.*]] = %[[B0]] to %[[B1]] step %[[C1]] {
// CHECK-TYPE0:   %[[IND0:.*]] = load %{{.*}}[%[[I]]] : memref<?xi64>
// CHECK-TYPE0:   %[[INDC:.*]] = index_cast %[[IND0]] : i64 to index
// CHECK-TYPE0:   %[[VAL0:.*]] = load %{{.*}}[%[[I]]] : memref<?xf64>
// CHECK-TYPE0:   %[[VAL1:.*]] = load %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK-TYPE0:   %[[MUL:.*]] = mulf %[[VAL0]], %[[VAL1]] : f64
// CHECK-TYPE0:   store %[[MUL]], %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK-TYPE0: }

// CHECK-TYPE1-LABEL: func @mul_dd(
// CHECK-TYPE1: %[[C0:.*]] = constant 0 : index
// CHECK-TYPE1: %[[C1:.*]] = constant 1 : index
// CHECK-TYPE1: %[[P0:.*]] = load %{{.*}}[%[[C0]]] : memref<?xi64>
// CHECK-TYPE1: %[[B0:.*]] = index_cast %[[P0]] : i64 to index
// CHECK-TYPE1: %[[P1:.*]] = load %{{.*}}[%[[C1]]] : memref<?xi64>
// CHECK-TYPE1: %[[B1:.*]] = index_cast %[[P1]] : i64 to index
// CHECK-TYPE1: scf.for %[[I:.*]] = %[[B0]] to %[[B1]] step %[[C1]] {
// CHECK-TYPE1:   %[[IND0:.*]] = load %{{.*}}[%[[I]]] : memref<?xi32>
// CHECK-TYPE1:   %[[INDC:.*]] = index_cast %[[IND0]] : i32 to index
// CHECK-TYPE1:   %[[VAL0:.*]] = load %{{.*}}[%[[I]]] : memref<?xf64>
// CHECK-TYPE1:   %[[VAL1:.*]] = load %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK-TYPE1:   %[[MUL:.*]] = mulf %[[VAL0]], %[[VAL1]] : f64
// CHECK-TYPE1:   store %[[MUL]], %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK-TYPE1: }

// CHECK-TYPE2-LABEL: func @mul_dd(
// CHECK-TYPE2: %[[C0:.*]] = constant 0 : index
// CHECK-TYPE2: %[[C1:.*]] = constant 1 : index
// CHECK-TYPE2: %[[P0:.*]] = load %{{.*}}[%[[C0]]] : memref<?xi32>
// CHECK-TYPE2: %[[B0:.*]] = index_cast %[[P0]] : i32 to index
// CHECK-TYPE2: %[[P1:.*]] = load %{{.*}}[%[[C1]]] : memref<?xi32>
// CHECK-TYPE2: %[[B1:.*]] = index_cast %[[P1]] : i32 to index
// CHECK-TYPE2: scf.for %[[I:.*]] = %[[B0]] to %[[B1]] step %[[C1]] {
// CHECK-TYPE2:   %[[IND0:.*]] = load %{{.*}}[%[[I]]] : memref<?xi64>
// CHECK-TYPE2:   %[[INDC:.*]] = index_cast %[[IND0]] : i64 to index
// CHECK-TYPE2:   %[[VAL0:.*]] = load %{{.*}}[%[[I]]] : memref<?xf64>
// CHECK-TYPE2:   %[[VAL1:.*]] = load %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK-TYPE2:   %[[MUL:.*]] = mulf %[[VAL0]], %[[VAL1]] : f64
// CHECK-TYPE2:   store %[[MUL]], %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK-TYPE2: }

// CHECK-TYPE3-LABEL: func @mul_dd(
// CHECK-TYPE3: %[[C0:.*]] = constant 0 : index
// CHECK-TYPE3: %[[C1:.*]] = constant 1 : index
// CHECK-TYPE3: %[[P0:.*]] = load %{{.*}}[%[[C0]]] : memref<?xi32>
// CHECK-TYPE3: %[[B0:.*]] = index_cast %[[P0]] : i32 to index
// CHECK-TYPE3: %[[P1:.*]] = load %{{.*}}[%[[C1]]] : memref<?xi32>
// CHECK-TYPE3: %[[B1:.*]] = index_cast %[[P1]] : i32 to index
// CHECK-TYPE3: scf.for %[[I:.*]] = %[[B0]] to %[[B1]] step %[[C1]] {
// CHECK-TYPE3:   %[[IND0:.*]] = load %{{.*}}[%[[I]]] : memref<?xi32>
// CHECK-TYPE3:   %[[INDC:.*]] = index_cast %[[IND0]] : i32 to index
// CHECK-TYPE3:   %[[VAL0:.*]] = load %{{.*}}[%[[I]]] : memref<?xf64>
// CHECK-TYPE3:   %[[VAL1:.*]] = load %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK-TYPE3:   %[[MUL:.*]] = mulf %[[VAL0]], %[[VAL1]] : f64
// CHECK-TYPE3:   store %[[MUL]], %{{.*}}[%[[INDC]]] : memref<32xf64>
// CHECK-TYPE3: }

func @mul_dd(%arga: tensor<32xf64>, %argb: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait_mul_1d
     ins(%arga, %argb: tensor<32xf64>, tensor<32xf64>)
    outs(%arga : tensor<32xf64>) {
      ^bb(%a: f64, %b: f64, %s: f64):
        %0 = mulf %a, %b  : f64
        linalg.yield %0 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

