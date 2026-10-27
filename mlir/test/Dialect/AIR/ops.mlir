// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @matrices
func.func @matrices(%m: memref<?x?xf16, strided<[?, 1]>, 1>, %i: index) {
  // CHECK: air.simdgroup_matrix_load %{{.*}}[%{{.*}}, %{{.*}}] transpose : memref<?x?xf16, strided<[?, 1]>, 1> -> vector<64xf16>
  %a = air.simdgroup_matrix_load %m[%i, %i] transpose : memref<?x?xf16, strided<[?, 1]>, 1> -> vector<64xf16>
  %x = arith.constant 0.5 : f16
  // CHECK: air.simdgroup_matrix_fill %{{.*}} : vector<64xf16>
  %c = air.simdgroup_matrix_fill %x : vector<64xf16>
  // CHECK: air.simdgroup_matrix_multiply_accumulate %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf16>
  %d = air.simdgroup_matrix_multiply_accumulate %a, %a, %c : vector<64xf16>
  %y = arith.constant 0.5 : f32
  %c32 = air.simdgroup_matrix_fill %y : vector<64xf32>
  // CHECK: air.simdgroup_matrix_multiply_accumulate %{{.*}}, %{{.*}}, %{{.*}} : vector<64xf16>, vector<64xf32>
  %e = air.simdgroup_matrix_multiply_accumulate %a, %a, %c32 : vector<64xf16>, vector<64xf32>
  // CHECK: air.simdgroup_matrix_store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<64xf16> -> memref<?x?xf16, strided<[?, 1]>, 1>
  air.simdgroup_matrix_store %d, %m[%i, %i] : vector<64xf16> -> memref<?x?xf16, strided<[?, 1]>, 1>
  return
}
