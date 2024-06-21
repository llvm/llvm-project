// RUN: mlir-opt %s -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @maskedstore16(%base: memref<?xf32>,
                    %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %base[%c0], %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

func.func @maskedstore16_at8(%base: memref<?xf32>,
                        %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c8 = arith.constant 8: index
  vector.maskedstore %base[%c8], %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

func.func @printmem16(%A: memref<?xf32>) {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c16 = arith.constant 16: index
  %z = arith.constant 0.0: f32
  %m = vector.broadcast %z : f32 to vector<16xf32>
  %mem = scf.for %i = %c0 to %c16 step %c1
    iter_args(%m_iter = %m) -> (vector<16xf32>) {
    %c = memref.load %A[%i] : memref<?xf32>
    %i32 = arith.index_cast %i : index to i32
    %m_new = vector.insertelement %c, %m_iter[%i32 : i32] : vector<16xf32>
    scf.yield %m_new : vector<16xf32>
  }
  vector.print %mem : vector<16xf32>
  return
}

func.func @entry() {
  // Set up memory.
  %f0 = arith.constant 0.0: f32
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c16 = arith.constant 16: index
  %A = memref.alloc(%c16) : memref<?xf32>
  scf.for %i = %c0 to %c16 step %c1 {
    memref.store %f0, %A[%i] : memref<?xf32>
  }

  // Set up value vector.
  %v = vector.broadcast %f0 : f32 to vector<16xf32>
  %val = scf.for %i = %c0 to %c16 step %c1
    iter_args(%v_iter = %v) -> (vector<16xf32>) {
    %i32 = arith.index_cast %i : index to i32
    %fi = arith.sitofp %i32 : i32 to f32
    %v_new = vector.insertelement %fi, %v_iter[%i32 : i32] : vector<16xf32>
    scf.yield %v_new : vector<16xf32>
  }

  // Set up masks.
  %t = arith.constant 1: i1
  %none = vector.constant_mask [0] : vector<16xi1>
  %some = vector.constant_mask [8] : vector<16xi1>
  %more = vector.insert %t, %some[13] : i1 into vector<16xi1>
  %all = vector.constant_mask [16] : vector<16xi1>

  //
  // Masked store tests.
  //

  vector.print %val : vector<16xf32>
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 )

  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )

  call @maskedstore16(%A, %none, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )

  call @maskedstore16(%A, %some, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0 )

  call @maskedstore16(%A, %more, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 13, 0, 0 )

  call @maskedstore16(%A, %all, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 )

  call @maskedstore16_at8(%A, %some, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 )

  memref.dealloc %A : memref<?xf32>
  return
}
