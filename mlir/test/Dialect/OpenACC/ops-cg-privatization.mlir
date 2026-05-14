// RUN: mlir-opt -split-input-file %s | FileCheck %s --check-prefixes=CHECK
// RUN: mlir-opt -split-input-file %s | mlir-opt -split-input-file | FileCheck %s --check-prefixes=CHECK
// RUN: mlir-opt -split-input-file -mlir-print-op-generic %s | mlir-opt -split-input-file | FileCheck %s --check-prefixes=CHECK

// -----

// CHECK-LABEL: func @privatize_static_scalar
func.func @privatize_static_scalar() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %h = acc.privatize : () -> !acc.private_type<memref<f32>>
  scf.parallel (%tx) = (%c0) to (%c128) step (%c1) {
    %loc = acc.private_local %h : (!acc.private_type<memref<f32>>) -> memref<f32>
    %z = arith.constant 0.0 : f32
    memref.store %z, %loc[] : memref<f32>
    scf.reduce
  } {acc.par_dims = #acc<par_dims[thread_x]>}
  return
}
// CHECK-DAG: %[[H:.*]] = acc.privatize
// CHECK-SAME: () -> !acc.private_type<memref<f32>>
// CHECK: scf.parallel
// CHECK: %{{.*}} = acc.private_local %[[H]] : (!acc.private_type<memref<f32>>) -> memref<f32>
// CHECK: memref.store
// CHECK: } {acc.par_dims = #acc<par_dims[thread_x]>}

// -----

// CHECK-LABEL: func @privatize_with_par_dims_attr
func.func @privatize_with_par_dims_attr() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %h = acc.privatize [#acc<par_dims[block_x, thread_x]>] : () -> !acc.private_type<memref<i32>>
  scf.parallel (%bx, %tx) = (%c0, %c0) to (%c8, %c128) step (%c1, %c1) {
    %loc = acc.private_local %h : (!acc.private_type<memref<i32>>) -> memref<i32>
    %z = arith.constant 0 : i32
    memref.store %z, %loc[] : memref<i32>
    scf.reduce
  } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}
// CHECK-DAG: %[[H2:.*]] = acc.privatize [#acc<par_dims[block_x, thread_x]>] : () -> !acc.private_type<memref<i32>>
// CHECK: %{{.*}} = acc.private_local %[[H2]] : (!acc.private_type<memref<i32>>) -> memref<i32>

// -----

// CHECK-LABEL: func @privatize_dynamic_row
func.func @privatize_dynamic_row(%n : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %h = acc.privatize(%n) : (index) -> !acc.private_type<memref<?xf32>>
  scf.parallel (%tx) = (%c0) to (%c32) step (%c1) {
    %row = acc.private_local %h : (!acc.private_type<memref<?xf32>>) -> memref<?xf32>
    %c0_f32 = arith.constant 0.0 : f32
    memref.store %c0_f32, %row[%c0] : memref<?xf32>
    scf.reduce
  } {acc.par_dims = #acc<par_dims[thread_x]>}
  return
}
// CHECK: acc.privatize(%{{.*}}) : (index) -> !acc.private_type<memref<?xf32>>
// CHECK: %{{.*}} = acc.private_local %{{.*}} : (!acc.private_type<memref<?xf32>>) -> memref<?xf32>

// -----

// CHECK-LABEL: func @privatize_dynamic_and_par_dims
func.func @privatize_dynamic_and_par_dims(%rows : index, %cols : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %h = acc.privatize(%rows, %cols) [#acc<par_dims[block_x, thread_x]>] : (index, index) -> !acc.private_type<memref<?x?xi32>>
  scf.parallel (%bx, %tx) = (%c0, %c0) to (%c4, %c16) step (%c1, %c1) {
    %tile = acc.private_local %h : (!acc.private_type<memref<?x?xi32>>) -> memref<?x?xi32>
    %z = arith.constant 0 : i32
    memref.store %z, %tile[%c0, %c0] : memref<?x?xi32>
    scf.reduce
  } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}
// CHECK: acc.privatize(%{{.*}}, %{{.*}}) [#acc<par_dims[block_x, thread_x]>] : (index, index) -> !acc.private_type<memref<?x?xi32>>

// -----

// CHECK-LABEL: func @privatize_inside_compute_region
func.func @privatize_inside_compute_region(%data : memref<64xf32>) {
  %copy = acc.copyin varPtr(%data : memref<64xf32>) -> memref<64xf32>
  acc.kernel_environment dataOperands(%copy : memref<64xf32>) {
    %c64_kw = arith.constant 64 : index
    %w = acc.par_width %c64_kw {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%lw = %w) ins(%d = %copy) : (memref<64xf32>) {
      %priv = acc.privatize [#acc<par_dims[thread_x]>] : () -> !acc.private_type<memref<f32>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %ub = arith.constant 64 : index
      scf.parallel (%iv) = (%c0) to (%ub) step (%c1) {
        %acc = acc.private_local %priv : (!acc.private_type<memref<f32>>) -> memref<f32>
        %v = memref.load %d[%iv] : memref<64xf32>
        memref.store %v, %acc[] : memref<f32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  acc.delete accPtr(%copy : memref<64xf32>)
  return
}
// CHECK: acc.kernel_environment
// CHECK: acc.compute_region
// CHECK: acc.privatize [#acc<par_dims[thread_x]>] : () -> !acc.private_type<memref<f32>>
// CHECK: acc.private_local
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<64xf32>
// CHECK: } {origin = "acc.parallel"}
