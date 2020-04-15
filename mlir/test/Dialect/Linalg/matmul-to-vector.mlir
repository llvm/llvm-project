// RUN: mlir-opt %s -linalg-matmul-to-vector | FileCheck %s

func @matmul_perm(%A: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                  %B: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
                  %C: memref<1584x1584xf32, offset: 0, strides: [1584, 1]>) {
  linalg.matmul(%A, %B, %C) {__internal_linalg_transform__ = "__with_perm__"} :
    memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
    memref<1584x1584xf32, offset: 0, strides: [1584, 1]>,
    memref<1584x1584xf32, offset: 0, strides: [1584, 1]>
  return
}

// CHECK-LABEL:func @matmul_perm
//      CHECK:        vector.contract
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: : vector<8x16xf32>, vector<16x12xf32> into vector<8x12xf32>
