// RUN: mlir-opt %s -test-linalg-transform-patterns=test-matmul-to-vector-patterns-tile-1d | FileCheck %s -check-prefix=CHECK-1D
// RUN: mlir-opt %s -test-linalg-transform-patterns=test-matmul-to-vector-patterns-tile-2d | FileCheck %s -check-prefix=CHECK-2D

func @matmul(%A: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                  %B: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                  %C: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>) {
  linalg.matmul {__internal_linalg_transform__ = "START"}
    ins(%A, %B: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                memref<1584x1584xf32, offset: 0, strides: [1584, 1]>)
   outs(%C: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>)
  return
}

// CHECK-1D-LABEL:func @matmul
//      CHECK-1D: vector.transfer_write {{.*}} : vector<8x16xf32>, memref<8x16xf32>
//      CHECK-1D: vector.transfer_write {{.*}} : vector<16x12xf32>, memref<16x12xf32>
//      CHECK-1D: vector.transfer_write {{.*}} : vector<8x12xf32>, memref<8x12xf32>
//
//      CHECK-1D: vector.transfer_read {{.*}} : memref<8x16xf32, #{{.*}}>, vector<8x16xf32>
//      CHECK-1D: vector.transfer_write {{.*}} : vector<8x16xf32>, memref<8x16xf32>
//      CHECK-1D: vector.transfer_read {{.*}} : memref<16x12xf32, #{{.*}}>, vector<16x12xf32>
//      CHECK-1D: vector.transfer_write {{.*}} : vector<16x12xf32>, memref<16x12xf32>
//      CHECK-1D: vector.transfer_read {{.*}} : memref<8x12xf32, #{{.*}}>, vector<8x12xf32>
//      CHECK-1D: vector.transfer_write {{.*}} : vector<8x12xf32>, memref<8x12xf32>
//
//      CHECK-1D: vector.contract
// CHECK-1D-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-1D-SAME:   : vector<8x16xf32>, vector<16x12xf32> into vector<8x12xf32>
//
//      CHECK-1D: vector.transfer_read {{.*}} : memref<8x12xf32>, vector<8x12xf32>
//      CHECK-1D: vector.transfer_write {{.*}} : vector<8x12xf32>, memref<8x12xf32, #{{.*}}>

// CHECK-2D-LABEL:func @matmul
//      CHECK-2D: vector.transfer_write {{.*}} : vector<8x16xf32>, memref<8x16xf32>
//      CHECK-2D: vector.transfer_write {{.*}} : vector<16x12xf32>, memref<16x12xf32>
//      CHECK-2D: vector.transfer_write {{.*}} : vector<8x12xf32>, memref<8x12xf32>
//
//      CHECK-2D: linalg.copy
//      CHECK-2D: linalg.copy
//      CHECK-2D: linalg.copy
//
//      CHECK-2D: vector.contract
// CHECK-2D-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-2D-SAME:   : vector<8x16xf32>, vector<16x12xf32> into vector<8x12xf32>
//
//      CHECK-2D: linalg.copy
