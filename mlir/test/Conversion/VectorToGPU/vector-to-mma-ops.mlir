// RUN: mlir-opt %s -convert-vector-to-gpu -canonicalize | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul
//   CHECK-DAG:   %[[A:.+]] = gpu.subgroup_mma_load_matrix %{{.*}}[%{{.*}}, %{{.*}}] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
//   CHECK-DAG:   %[[B:.+]] = gpu.subgroup_mma_load_matrix %{{.*}}[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
//   CHECK-DAG:   %[[C:.+]] = gpu.subgroup_mma_load_matrix %{{.*}}[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
//       CHECK:   %[[D:.+]] = gpu.subgroup_mma_compute %[[A]], %[[B]], %[[C]] : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
//       CHECK:   gpu.subgroup_mma_store_matrix %[[D]], %{{.*}}[%{{.*}}, %{{.*}}] {leadDimension = 16 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16>
func @matmul(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<16x16xf16>) {
  %cst_0 = constant dense<0.000000e+00> : vector<16x16xf16>
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f16
  %A = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
  %B = vector.transfer_read %arg1[%c0, %c0], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
  %C = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<16x16xf16>
  return
}

// CHECK-LABEL: func @matmul_cst
//   CHECK-DAG:   %[[CST:.+]] = constant 0.000000e+00 : f16
//   CHECK-DAG:   %[[A:.+]] = gpu.subgroup_mma_load_matrix %{{.*}}[%{{.*}}, %{{.*}}] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
//   CHECK-DAG:   %[[B:.+]] = gpu.subgroup_mma_load_matrix %{{.*}}[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
//   CHECK-DAG:   %[[C:.+]] = gpu.subgroup_mma_constant_matrix %[[CST]] : !gpu.mma_matrix<16x16xf16, "COp">
//       CHECK:   %[[D:.+]] = gpu.subgroup_mma_compute %[[A]], %[[B]], %[[C]] : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
//       CHECK:   gpu.subgroup_mma_store_matrix %[[D]], %{{.*}}[%{{.*}}, %{{.*}}] {leadDimension = 16 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<16x16xf16>
func @matmul_cst(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<16x16xf16>) {
  %cst_0 = constant dense<0.000000e+00> : vector<16x16xf16>
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f16
  %A = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
  %B = vector.transfer_read %arg1[%c0, %c0], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %cst_0 : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<16x16xf16>
  return
}

// CHECK-LABEL: func @matmul_loop
//       CHECK:   %[[C:.+]] = gpu.subgroup_mma_load_matrix %{{.*}}[%{{.*}}, %{{.*}}] {leadDimension = 128 : index} : memref<128x128xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
//       CHECK:   %[[ACC:.+]] = scf.for {{.*}} iter_args(%[[ACC1:.+]] = %[[C]]) -> (!gpu.mma_matrix<16x16xf16, "COp">) {
//   CHECK-DAG:     %[[A:.+]] = gpu.subgroup_mma_load_matrix %{{.*}}[%{{.*}}, %{{.*}}] {leadDimension = 128 : index} : memref<128x128xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
//   CHECK-DAG:     %[[B:.+]] = gpu.subgroup_mma_load_matrix %{{.*}}[%{{.*}}, %{{.*}}] {leadDimension = 128 : index} : memref<128x128xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
//  CHECK-NEXT:     %[[D:.+]] = gpu.subgroup_mma_compute %[[A]], %[[B]], %[[ACC1]] : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
//  CHECK-NEXT:     scf.yield %[[D]] : !gpu.mma_matrix<16x16xf16, "COp">
//  CHECK-NEXT:   }
//  CHECK-NEXT:   gpu.subgroup_mma_store_matrix %[[ACC]], %{{.*}}[%{{.*}}, %{{.*}}] {leadDimension = 128 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<128x128xf16>
func @matmul_loop(%arg0: memref<128x128xf16>, %arg1: memref<128x128xf16>, %arg2: memref<128x128xf16>) {
  %c0 = constant 0 : index
  %c128 = constant 128 : index
  %c32 = constant 32 : index
  %cst = constant 0.000000e+00 : f16
  %C = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
  %14 = scf.for %arg17 = %c0 to %c128 step %c32 iter_args(%arg18 = %C) -> (vector<16x16xf16>) {
    %17 = vector.transfer_read %arg0[%c0, %arg17], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
    %18 = vector.transfer_read %arg1[%arg17, %c0], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
    %19 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %17, %18, %arg18 : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
    scf.yield %19 : vector<16x16xf16>
  }
  vector.transfer_write %14, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<128x128xf16>
  return
}
