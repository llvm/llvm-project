// RUN: mlir-opt %s -split-input-file -convert-mesh-to-mpi | FileCheck %s

// CHECK: mesh.mesh @mesh0
mesh.mesh @mesh0(shape = 2x2x4)

// -----

// CHECK-LABEL: func @update_halo
func.func @update_halo_1d(
    // CHECK-SAME: %[[ARG:.*]]: memref<12x12xi8>
    %arg0 : memref<12x12xi8>) {
  // CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: mesh.update_halo %[[ARG]] on @mesh0
  // CHECK-SAME: split_axes = {{\[\[}}0]]
  // CHECK-SAME: halo_sizes = [2, %c2_i64] : memref<12x12xi8>
  %c2 = arith.constant 2 : i64
  mesh.update_halo %arg0 on @mesh0 split_axes = [[0]]
    halo_sizes = [2, %c2] : memref<12x12xi8>
  return
}

func.func @update_halo_2d(
    // CHECK-SAME: %[[ARG:.*]]: memref<12x12xi8>
    %arg0 : memref<12x12xi8>) {
  %c2 = arith.constant 2 : i64
  // CHECK-NEXT: mesh.update_halo %[[ARG]] on @mesh0
  // CHECK-SAME: split_axes = {{\[\[}}0], [1]]
  // CHECK-SAME: halo_sizes = [2, 2, %[[C2]], 2]
  // CHECK-SAME: target_halo_sizes = [3, 3, 2, 2] : memref<12x12xi8>
  mesh.update_halo %arg0 on @mesh0 split_axes = [[0], [1]]
    halo_sizes = [2, 2, %c2, 2] target_halo_sizes = [3, 3, 2, 2]
    : memref<12x12xi8>
  return
}
