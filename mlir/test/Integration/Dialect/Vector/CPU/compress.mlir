// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | \
// RUN: FileCheck %s

//===----------------------------------------------------------------------===//
// @compress_16
//
// Insertion index is hard-coded to 0
//===----------------------------------------------------------------------===//
func.func @compress_16(%base: memref<?xf32>,
                 %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0: index
  vector.compressstore %base[%c0], %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

//===----------------------------------------------------------------------===//
// @compress_16_at_8
//
// Same as @compress_16, but the insertion index is hard-coded to 8 instead of 0
//===----------------------------------------------------------------------===//
func.func @compress_16_at_8(%base: memref<?xf32>,
                     %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c8 = arith.constant 8: index
  vector.compressstore %base[%c8], %mask, %value
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
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c16 = arith.constant 16: index
  %A = memref.alloc(%c16) : memref<?xf32>
  %z = arith.constant 0.0: f32
  %v = vector.broadcast %z : f32 to vector<16xf32>
  %value = scf.for %i = %c0 to %c16 step %c1
    iter_args(%v_iter = %v) -> (vector<16xf32>) {
    memref.store %z, %A[%i] : memref<?xf32>
    %i32 = arith.index_cast %i : index to i32
    %fi = arith.sitofp %i32 : i32 to f32
    %v_new = vector.insert %fi, %v_iter[%i] : f32 into vector<16xf32>
    scf.yield %v_new : vector<16xf32>
  }

  // Set up masks.
  %f = arith.constant 0: i1
  %t = arith.constant 1: i1
  // %none = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  %none = vector.constant_mask [0] : vector<16xi1>
  // %all = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  %all = vector.constant_mask [16] : vector<16xi1>
  // %some1 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  %some1 = vector.constant_mask [4] : vector<16xi1>
  %0 = vector.insert %f, %some1[0] : i1 into vector<16xi1>
  %1 = vector.insert %t, %0[7] : i1 into vector<16xi1>
  %2 = vector.insert %t, %1[11] : i1 into vector<16xi1>
  %3 = vector.insert %t, %2[13] : i1 into vector<16xi1>
  // %some2 = [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  %some2 = vector.insert %t, %3[15] : i1 into vector<16xi1>
  // %some3 = [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  %some3 = vector.insert %f, %some2[2] : i1 into vector<16xi1>

  //
  // Expanding load tests.
  //

  call @compress_16(%A, %none, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  call @compress_16(%A, %all, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

  call @compress_16(%A, %some3, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [1, 3, 7, 11, 13, 15, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

  call @compress_16(%A, %some2, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [1, 2, 3, 7, 11, 13, 15, 7, 8, 9, 10, 11, 12, 13, 14, 15]

  call @compress_16(%A, %some1, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [0, 1, 2, 3, 11, 13, 15, 7, 8, 9, 10, 11, 12, 13, 14, 15]

  call @compress_16_at_8(%A, %some1, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @print1DMemRef(%A) : (memref<?xf32>) -> ()
  // CHECK: [0, 1, 2, 3, 11, 13, 15, 7, 0, 1, 2, 3, 12, 13, 14, 15]

  memref.dealloc %A : memref<?xf32>
  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
