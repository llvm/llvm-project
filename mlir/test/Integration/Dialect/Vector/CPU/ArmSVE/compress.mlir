// REQUIRES: arm-emulator

/// End-to-end test for vector.compressstore for SVE

// In order to demonstrate the impact of using scalable vectors, vscale is set
// to 2 so that vector<[16]xi32> contains 32 rather than 16 elements at
// run-time
//
// Note that you can also tweak the size of vscale by passing this flag to
// QEMU:
//  * -cpu max,sve-max-vq=[1-16]
// (select the value between 1 and 16).

// DEFINE: %{compile} =  mlir-opt %s -test-lower-to-llvm -o %t
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e main -entry-point-result=void --march=aarch64 --mattr="+sve"\
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} &&  %{run} |  FileCheck %s

//===----------------------------------------------------------------------===//
// @compress_16
//
// The number of inserted elements is 16 x vscale. Insertion index is
// hard-coded to 0
//===----------------------------------------------------------------------===//
func.func @compress_16(%base: memref<?xi32>,
                 %mask: vector<[16]xi1>, %value: vector<[16]xi32>) {
  %c0 = arith.constant 0: index
  vector.compressstore %base[%c0], %mask, %value
    : memref<?xi32>, vector<[16]xi1>, vector<[16]xi32>
  return
}

//===----------------------------------------------------------------------===//
// @compress_16_at_8
//
// Same as @compress_16, but the insertion index is hard-coded to 8 instead of 0
//===----------------------------------------------------------------------===//
func.func @compress_16_at_8(%base: memref<?xi32>,
                     %mask: vector<[16]xi1>, %value: vector<[16]xi32>) {
  %c8 = arith.constant 8: index
  vector.compressstore %base[%c8], %mask, %value
    : memref<?xi32>, vector<[16]xi1>, vector<[16]xi32>
  return
}

//===----------------------------------------------------------------------===//
// @print1DMemRef
//
// TODO: Move to an utility file
//===----------------------------------------------------------------------===//
func.func @print1DMemRef(%ptr: memref<?xi32>) -> () {
  %cast = memref.cast %ptr:  memref<?xi32> to memref<*xi32>

  call @printMemrefI32(%cast): (memref<*xi32>) -> ()

  return
}

//===----------------------------------------------------------------------===//
// @reset_mem_i32
//
// Resets the input memory to 0.

// TODO: Create a run-time utility funcion.
//===----------------------------------------------------------------------===//
func.func @reset_mem_i32(%ptr: memref<?xi32>, %size: index) {
  %c0_idx = arith.constant 0: index
  %c0 = arith.constant 0: i32
  %step = arith.constant 1: index

  scf.for %i = %c0_idx to %size step %step {
    memref.store %c0, %ptr[%i] : memref<?xi32>
  }

  return
}

//===----------------------------------------------------------------------===//
// @main
//
// The main entry point - sets the value of vscale.
//===----------------------------------------------------------------------===//
func.func @main() {
  // Set vscale to 2 (vector width = 256). This will have identical effect to:
  //  * qemu-aarch64 -cpu max,sve-max-vq=2 (...)
  %c256 = arith.constant 256 : i32
  func.call @setArmVLBits(%c256) : (i32) -> ()

  // Run the tests.
  func.call @test() : () -> ()

  return
}

//===----------------------------------------------------------------------===//
// @test
//
// Set-up and run tests.
//===----------------------------------------------------------------------===//
func.func @test() {
  //
  // Shared constants.
  //
  %vs = vector.vscale

  //
  // Set up memory.
  //
  %c16 = arith.constant 16: index
  %vs_16 = arith.muli %vs, %c16 : index
  %A = memref.alloc(%vs_16) : memref<?xi32>
  call @reset_mem_i32(%A, %vs_16) : (memref<?xi32>, index) -> ()

  //
  // Set the input vector.
  //
  %value = vector.step : vector<[16]xi32>
  vector.print %value : vector<[16]xi32>

  //
  // Set up masks.
  //

  %f = arith.constant 0: i1
  %t = arith.constant 1: i1

  // %none = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 0]
  %none = vector.constant_mask [0] : vector<[16]xi1>

  // %all = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ..., 1]
  %all = vector.constant_mask [16] : vector<[16]xi1>
  vector.print %all : vector<[16]xi1>

  // %first_vscale_4 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ..., 0]
  %c4 = arith.constant 4 : index
  %vs_4 = arith.muli %vs, %c4 : index
  %first_vscale_4 = vector.create_mask %vs_4 : vector<[16]xi1>
  vector.print %first_vscale_4 : vector<[16]xi1>

  // %odd = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, ..., 0]
  %0 = vector.insert %t, %none[1] : i1 into vector<[16]xi1>
  %1 = vector.insert %t, %0[3] : i1 into vector<[16]xi1>
  %2 = vector.insert %t, %1[5] : i1 into vector<[16]xi1>
  %3 = vector.insert %t, %2[7] : i1 into vector<[16]xi1>
  %4 = vector.insert %t, %3[9] : i1 into vector<[16]xi1>
  %5 = vector.insert %t, %4[11] : i1 into vector<[16]xi1>
  %odd = vector.insert %t, %5[13] : i1 into vector<[16]xi1>

  // %even = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ..., 0]
  %6 = vector.insert %t, %none[0] : i1 into vector<[16]xi1>
  %7 = vector.insert %t, %6[2] : i1 into vector<[16]xi1>
  %8 = vector.insert %t, %7[4] : i1 into vector<[16]xi1>
  %9 = vector.insert %t, %8[6] : i1 into vector<[16]xi1>
  %10 = vector.insert %t, %9[8] : i1 into vector<[16]xi1>
  %11 = vector.insert %t, %10[10] : i1 into vector<[16]xi1>
  %even = vector.insert %t, %11[12] : i1 into vector<[16]xi1>


  //
  // Tests.
  //

  call @compress_16(%A, %none, %value)
    : (memref<?xi32>, vector<[16]xi1>, vector<[16]xi32>) -> ()
  call @print1DMemRef(%A) : (memref<?xi32>) -> ()
  // CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  /// (...)
  // CHECK-SAME: 0, 0]

  call @compress_16(%A, %first_vscale_4, %value)
    : (memref<?xi32>, vector<[16]xi1>, vector<[16]xi32>) -> ()
  call @print1DMemRef(%A) : (memref<?xi32>) -> ()
  // CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0,
  /// (...)
  // CHECK-SAME: 0, 0]

  call @compress_16(%A, %all, %value)
    : (memref<?xi32>, vector<[16]xi1>, vector<[16]xi32>) -> ()
  call @print1DMemRef(%A) : (memref<?xi32>) -> ()
  // CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
  /// (...)
  // CHECK-SAME: 30, 31]

  call @reset_mem_i32(%A, %vs_16) : (memref<?xi32>, index) -> ()
  call @compress_16_at_8(%A, %first_vscale_4, %value)
    : (memref<?xi32>, vector<[16]xi1>, vector<[16]xi32>) -> ()
  call @print1DMemRef(%A) : (memref<?xi32>) -> ()
  // CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0,
  /// (...)
  // CHECK-SAME: 0, 0]

  call @reset_mem_i32(%A, %vs_16) : (memref<?xi32>, index) -> ()
  call @compress_16(%A, %odd, %value)
    : (memref<?xi32>, vector<[16]xi1>, vector<[16]xi32>) -> ()
  call @print1DMemRef(%A) : (memref<?xi32>) -> ()
  // CHECK: [1,  3,  5,  7,  9,  11,  13,  0,  0,  0,  0,
  /// (...)
  // CHECK-SAME: 0, 0]

  call @reset_mem_i32(%A, %vs_16) : (memref<?xi32>, index) -> ()
  call @compress_16(%A, %even, %value)
    : (memref<?xi32>, vector<[16]xi1>, vector<[16]xi32>) -> ()
  call @print1DMemRef(%A) : (memref<?xi32>) -> ()
  // CHECK: [0,  2,  4,  6,  8,  10,  12,  0,  0,  0,  0,
  /// (...)
  // CHECK-SAME: 0, 0]

  memref.dealloc %A : memref<?xi32>
  return
}

func.func private @printMemrefI32(%ptr : memref<*xi32>)
func.func private @setArmVLBits(%bits : i32)
