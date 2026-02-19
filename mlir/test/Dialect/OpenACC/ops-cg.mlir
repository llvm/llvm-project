// RUN: mlir-opt -split-input-file %s | FileCheck %s --check-prefixes=CHECK
// Verify the printed output can be parsed.
// RUN: mlir-opt -split-input-file %s | mlir-opt -split-input-file | FileCheck %s --check-prefixes=CHECK
// Verify the generic form can be parsed.
// RUN: mlir-opt -split-input-file -mlir-print-op-generic %s | mlir-opt -split-input-file | FileCheck %s --check-prefixes=CHECK

// -----

// CHECK-LABEL: func @par_dim_sequential
func.func @par_dim_sequential() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%iv) = (%c0) to (%c4) step (%c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[sequential]>}
  return
}
// CHECK: scf.parallel
// CHECK: } {acc.par_dims = #acc<par_dims[sequential]>}

// -----

// CHECK-LABEL: func @par_dim_single_thread_x
func.func @par_dim_single_thread_x() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%iv) = (%c0) to (%c128) step (%c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[thread_x]>}
  return
}
// CHECK: scf.parallel
// CHECK: } {acc.par_dims = #acc<par_dims[thread_x]>}

// -----

// CHECK-LABEL: func @par_dims_block_thread
func.func @par_dims_block_thread() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c8, %c128) step (%c1, %c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
  return
}
// CHECK: scf.parallel
// CHECK: } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}

// -----

// All GPU parallel dimensions (par_dim values) in par_dims list
func.func @par_dims_all_dims() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%iv) = (%c0) to (%c1) step (%c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[block_x, block_y, block_z, thread_x, thread_y, thread_z]>}
  return
}
// CHECK: acc.par_dims = #acc<par_dims[block_x, block_y, block_z, thread_x, thread_y, thread_z]>

// -----

// 2D grid: block_y and thread_y
func.func @par_dims_2d_grid() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c4, %c32) step (%c1, %c1) {
    scf.reduce
  } {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
  return
}
// CHECK: acc.par_dims = #acc<par_dims[block_y, thread_y]>
