// RUN: mlir-opt %s -test-linalg-codegen-strategy="tile-sizes=2,4,8 vectorize vectorize-contraction-to=matrixintrinsics unroll-vector-transfers=true" | FileCheck %s
// RUN: mlir-opt %s -test-linalg-codegen-strategy="tile-sizes=16,32,64 promote promote-full-tile-pad register-tile-sizes=2,4,8 vectorize vectorize-contraction-to=outerproduct split-transfers=true unroll-vector-transfers=false" | FileCheck %s --check-prefix=OUTER

// CHECK-LABEL: func @matmul(
// OUTER-LABEL: func @matmul(
func @matmul(%A: memref<1584x1584xf32>, %B: memref<1584x1584xf32>, %C: memref<1584x1584xf32>) {
  linalg.matmul
   ins(%A, %B: memref<1584x1584xf32>, memref<1584x1584xf32>)
   outs(%C: memref<1584x1584xf32>)

  //      CHECK: vector.matrix_multiply
  // CHECK-SAME: {lhs_columns = 8 : i32, lhs_rows = 2 : i32, rhs_columns = 4 : i32}
  // CHECK-SAME: (vector<16xf32>, vector<32xf32>) -> vector<8xf32>

  // OUTER: vector.outerproduct {{.*}} : vector<2xf32>, vector<4xf32>
  return
}

