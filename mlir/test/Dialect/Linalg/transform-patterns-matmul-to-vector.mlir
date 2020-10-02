// RUN: mlir-opt %s -test-linalg-transform-patterns=test-matmul-to-vector-patterns-tile-1d | FileCheck %s
// RUN: mlir-opt %s -test-linalg-transform-patterns=test-matmul-to-vector-patterns-tile-2d | FileCheck %s
// RUN: mlir-opt %s -test-linalg-transform-patterns=test-contraction-to-vector-patterns | FileCheck %s --check-prefix=VECTOR-CONTRACTION

func @matmul(%A: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                  %B: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                  %C: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>) {
  linalg.matmul {__internal_linalg_transform__ = "START"}
    ins(%A, %B: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                memref<1584x1584xf32, offset: 0, strides: [1584, 1]>)
   outs(%C: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>)
  return
}

// CHECK-LABEL:func @matmul
//      CHECK: store {{.*}}[] : memref<vector<8x16xf32>>
//      CHECK: store {{.*}}[] : memref<vector<16x12xf32>>
//      CHECK: store {{.*}}[] : memref<vector<8x12xf32>>
//
//      CHECK: linalg.copy
//      CHECK: linalg.copy
//      CHECK: linalg.copy
//
//      CHECK: vector.contract
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:   : vector<8x16xf32>, vector<16x12xf32> into vector<8x12xf32>
//
//      CHECK: linalg.copy

// VECTOR-CONTRACTION-LABEL: contraction_dot
func @contraction_dot(%A: memref<1584xf32>, %B: memref<1584xf32>, %C: memref<f32>) {
  // VECTOR-CONTRACTION: vector.contract
  // VECTOR-CONTRACTION-SAME: vector<1584xf32>, vector<1584xf32> into f32
  linalg.dot ins(%A, %B: memref<1584xf32>, memref<1584xf32>)
            outs(%C: memref<f32>)
  return
}

// VECTOR-CONTRACTION-LABEL: contraction_matvec
func @contraction_matvec(%A: memref<1584x1584xf32>, %B: memref<1584xf32>, %C: memref<1584xf32>) {
  // VECTOR-CONTRACTION: vector.contract
  // VECTOR-CONTRACTION-SAME: vector<1584x1584xf32>, vector<1584xf32> into vector<1584xf32>
  linalg.matvec ins(%A, %B: memref<1584x1584xf32>, memref<1584xf32>)
            outs(%C: memref<1584xf32>)
  return
}

// VECTOR-CONTRACTION-LABEL: contraction_matmul
func @contraction_matmul(%A: memref<1584x1584xf32>, %B: memref<1584x1584xf32>, %C: memref<1584x1584xf32>) {
  // VECTOR-CONTRACTION: vector.contract
  // VECTOR-CONTRACTION-SAME: vector<1584x1584xf32>, vector<1584x1584xf32> into vector<1584x1584xf32>
  linalg.matmul ins(%A, %B: memref<1584x1584xf32>, memref<1584x1584xf32>)
            outs(%C: memref<1584x1584xf32>)
  return
}

// VECTOR-CONTRACTION-LABEL: contraction_batch_matmul
func @contraction_batch_matmul(%A: memref<1584x1584x1584xf32>, %B: memref<1584x1584x1584xf32>, %C: memref<1584x1584x1584xf32>) {
  // VECTOR-CONTRACTION: vector.contract
  // VECTOR-CONTRACTION-SAME: vector<1584x1584x1584xf32>, vector<1584x1584x1584xf32> into vector<1584x1584x1584xf32>
  linalg.batch_matmul
    ins(%A, %B: memref<1584x1584x1584xf32>, memref<1584x1584x1584xf32>)
   outs(%C: memref<1584x1584x1584xf32>)
  return
}
