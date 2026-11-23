// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | \
// RUN: FileCheck %s

//===----------------------------------------------------------------------===//
// @maskedstore_16
//
// Store 16 elements. Insertion index is hard-coded to 0
//===----------------------------------------------------------------------===//
func.func @maskedstore_16(%base: memref<?xf32>,
                    %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %base[%c0], %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

//===----------------------------------------------------------------------===//
// @maskedstore_16_at_8
//
// Same as @maskedstore_16, but the insertion index is hard-coded to 8 instead of 0
//===----------------------------------------------------------------------===//
func.func @maskedstore_16_at_8(%base: memref<?xf32>,
                        %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c8 = arith.constant 8: index
  vector.maskedstore %base[%c8], %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

//===----------------------------------------------------------------------===//
// @print1DMemRef
//
// TODO: Move to an utility file
//===----------------------------------------------------------------------===//
func.func @print1DMemRef(%ptr: memref<?xf32>) -> () {
  %cast = memref.cast %ptr:  memref<?xf32> to memref<*xf32>

  call @printMemrefF32(%cast): (memref<*xf32>) -> ()

  return
}

//===----------------------------------------------------------------------===//
// @main
//
// The main entry point.
//===----------------------------------------------------------------------===//
func.func @main() {
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
    %v_new = vector.insert %fi, %v_iter[%i] : f32 into vector<16xf32>
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

  call @print1DMemRef(%A): (memref<?xf32>) -> ()
  // CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  call @maskedstore_16(%A, %none, %val)
  : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  call @maskedstore_16(%A, %some, %val)
  : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0]

  call @maskedstore_16(%A, %more, %val)
  : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 13, 0, 0]

  call @maskedstore_16(%A, %all, %val)
  : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

  call @maskedstore_16_at_8(%A, %some, %val)
  : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]

  memref.dealloc %A : memref<?xf32>
  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
