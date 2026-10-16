// RUN: mlir-opt %s -convert-vector-to-scf | FileCheck %s

// Allocations for transfers in serial loops belong to the enclosing
// automatic-allocation scope, not to the loop that contains the transfer.
// CHECK-LABEL: func.func @serial_loop(
// CHECK-NOT: scf.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK-NOT: scf.for
// CHECK: memref.alloca() : memref<vector<2x2xi1>>
// CHECK-NOT: scf.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK-NOT: scf.for
// CHECK: memref.alloca() : memref<vector<2x2xi1>>
// CHECK-NOT: scf.for
// CHECK: scf.for
// CHECK: vector.create_mask
// CHECK: memref.store {{.*}} : memref<vector<2x2xi1>>
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @serial_loop(%source: memref<?x?xf32>, %dest: memref<?x?xf32>,
                       %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  scf.for %i = %lb to %ub step %step {
    %mask = vector.create_mask %i, %i : vector<2x2xi1>
    %value = vector.transfer_read %source[%i, %c0], %pad, %mask
        {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x2xf32>
    vector.transfer_write %value, %dest[%i, %c0], %mask
        {in_bounds = [true, true]} : vector<2x2xf32>, memref<?x?xf32>
  }
  return
}

// The scope search also skips non-allocation-scope operations between serial
// loops. Both allocations must precede the outer loop.
// CHECK-LABEL: func.func @mixed_loop(
// CHECK-NOT: scf.for
// CHECK-NOT: affine.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK-NOT: scf.for
// CHECK-NOT: affine.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK: scf.for
// CHECK: scf.if
// CHECK: affine.for
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @mixed_loop(%source: memref<?x?xf32>, %dest: memref<?x?xf32>,
                      %rows: index, %enabled: i1) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %pad = arith.constant 0.0 : f32
  scf.for %i = %c0 to %rows step %c2 {
    scf.if %enabled {
      affine.for %j = 0 to 4 {
        %value = vector.transfer_read %source[%i, %j], %pad
            {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x2xf32>
        vector.transfer_write %value, %dest[%i, %j]
            {in_bounds = [true, true]} : vector<2x2xf32>, memref<?x?xf32>
      }
    }
  }
  return
}

// Parallel allocation scopes are preserved. The scratch buffers remain
// private to each parallel iteration while moving out of the nested serial
// loop.
// CHECK-LABEL: func.func @parallel_loop(
// CHECK-NOT: memref.alloca
// CHECK: scf.parallel
// CHECK-NOT: scf.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK-NOT: scf.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @parallel_loop(%source: memref<?x?xf32>, %dest: memref<?x?xf32>,
                         %threads: index, %rows: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %pad = arith.constant 0.0 : f32
  scf.parallel (%thread) = (%c0) to (%threads) step (%c1) {
    %row = arith.muli %thread, %c2 : index
    scf.for %i = %c0 to %rows step %c2 {
      %value = vector.transfer_read %source[%row, %i], %pad
          {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x2xf32>
      vector.transfer_write %value, %dest[%row, %i]
          {in_bounds = [true, true]} : vector<2x2xf32>, memref<?x?xf32>
    }
    scf.reduce
  }
  return
}

// scf.forall is another parallel allocation scope and must not be treated as
// a serial loop merely because it implements LoopLikeOpInterface.
// CHECK-LABEL: func.func @forall_loop(
// CHECK-NOT: memref.alloca
// CHECK: scf.forall
// CHECK-NOT: affine.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK-NOT: affine.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK: affine.for
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @forall_loop(%source: memref<?x?xf32>, %dest: memref<?x?xf32>,
                       %threads: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %pad = arith.constant 0.0 : f32
  scf.forall (%thread) in (%threads) {
    %row = arith.muli %thread, %c2 : index
    affine.for %i = 0 to 4 {
      %value = vector.transfer_read %source[%row, %i], %pad
          {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x2xf32>
      vector.transfer_write %value, %dest[%row, %i]
          {in_bounds = [true, true]} : vector<2x2xf32>, memref<?x?xf32>
    }
  }
  return
}

// An explicit allocation scope remains the lifetime boundary even when it is
// nested in a serial loop.
// CHECK-LABEL: func.func @explicit_scope(
// CHECK-NOT: memref.alloca
// CHECK: scf.for
// CHECK: memref.alloca_scope
// CHECK-NOT: scf.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK-NOT: scf.for
// CHECK: memref.alloca() : memref<vector<2x2xf32>>
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @explicit_scope(%source: memref<?x?xf32>, %dest: memref<?x?xf32>,
                          %rows: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pad = arith.constant 0.0 : f32
  scf.for %i = %c0 to %rows step %c1 {
    memref.alloca_scope {
      %value = vector.transfer_read %source[%i, %c0], %pad
          {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x2xf32>
      vector.transfer_write %value, %dest[%i, %c0]
          {in_bounds = [true, true]} : vector<2x2xf32>, memref<?x?xf32>
    }
  }
  return
}
