// RUN: mlir-opt -convert-scf-to-openmp='automatic-num-threads' -split-input-file %s | FileCheck %s

func.func @automatic(%arg0: memref<100xf32>) -> memref<100xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<100xf32>
  // CHECK: %[[NTH:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[NTH]] : i32)
  scf.parallel (%arg1) = (%c0) to (%c8) step (%c1) {
    %0 = memref.load %alloc[%arg1] : memref<100xf32>
    %1 = arith.addf %0, %cst : f32
    memref.store %1, %alloc[%arg1] : memref<100xf32>
    scf.reduce
  }
  return %alloc : memref<100xf32>
}

// -----
func.func @automatic_multiple_ub(%arg0: memref<100x100x100xf32>) -> memref<100x100x100xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<100x100x100xf32>
  // CHECK: %[[NTH:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[NTH]] : i32)
  scf.parallel (%arg1, %arg2, %arg3) = (%c0, %c0, %c0) to (%c8, %c2, %c2) step (%c1, %c1, %c1) {
    %0 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<100x100x100xf32>
    %1 = arith.addf %0, %cst : f32
    memref.store %1, %alloc[%arg1, %arg2, %arg3] : memref<100x100x100xf32>
    scf.reduce
  }
  return %alloc : memref<100x100x100xf32>
}

// -----
func.func @automatic_nonzero_lb(%arg0: memref<100xf32>) -> memref<100xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<100xf32>
  // CHECK: %[[NTH:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[NTH]] : i32)
  scf.parallel (%arg1) = (%c4) to (%c8) step (%c1) {
    %0 = memref.load %alloc[%arg1] : memref<100xf32>
    %1 = arith.addf %0, %cst : f32
    memref.store %1, %alloc[%arg1] : memref<100xf32>
    scf.reduce
  }
  return %alloc : memref<100xf32>
}

// -----
func.func @automatic_steps(%arg0: memref<100x100x100xf32>) -> memref<100x100x100xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c12 = arith.constant 12 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<100x100x100xf32>
  // CHECK: %[[NTH:.*]] = llvm.mlir.constant(12 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[NTH]] : i32)
  scf.parallel (%arg1, %arg2, %arg3) = (%c4, %c8, %c12) to (%c16, %c16, %c16) step (%c4, %c4, %c2) {
    %0 = memref.load %alloc[%arg1, %arg2, %arg3] : memref<100x100x100xf32>
    %1 = arith.addf %0, %cst : f32
    memref.store %1, %alloc[%arg1, %arg2, %arg3] : memref<100x100x100xf32>
    scf.reduce
  }
  return %alloc : memref<100x100x100xf32>
}
