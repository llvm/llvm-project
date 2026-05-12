// REQUIRES: arm-emulator

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --arm-sve-legalize-vector-storage --convert-vector-to-scf --convert-scf-to-cf  --convert-vector-to-llvm='enable-arm-sve' \
// DEFINE:   --expand-strided-metadata    --lower-affine --convert-to-llvm --finalize-memref-to-llvm  --reconcile-unrealized-casts \
// DEFINE: -o %t

// DEFINE: %{entry_point} = main

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void  --march=aarch64 --mattr="+sve" \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

// Test the transfer_read with vector type with a non-trailing scalable
// dimension as transformed by the pattern LegalizeTransferRead.

func.func @transfer_read_scalable_non_trailing(%M : memref<?x8xi8>) attributes {no_inline} {
  // Read an LLVM-illegal vector
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8
  %A = vector.transfer_read %M[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<?x8xi8>, vector<[4]x8xi8>

  // Print the vector, for verification.
  %B = vector.shape_cast %A : vector<[4]x8xi8> to vector<[32]xi8>
  func.call @printVec(%B) : (vector<[32]xi8>) -> ()

  return
}

func.func @main() {

  %c0 = arith.constant 0 : index

// Prepare an 8x8 buffer with test data. The test performs two reads
// of a [4]x8 vector from the buffer. One read, with vector length 128 bits,
// reads the first half the buffer. The other read, with vector length
// 256 bits, reads the entire buffer.
  %T = arith.constant dense<[[11, 12, 13, 14, 15, 16, 17, 18],
                             [21, 22, 23, 24, 25, 26, 27, 28],
                             [31, 32, 33, 34, 35, 36, 37, 38],
                             [41, 42, 43, 44, 45, 46, 47, 48],
                             [51, 52, 53, 54, 55, 56, 57, 58],
                             [61, 62, 63, 64, 65, 66, 67, 68],
                             [71, 72, 73, 74, 75, 76, 77, 78],
                             [81, 82, 83, 84, 85, 86, 87, 88]]> : vector<8x8xi8>

  %M = memref.alloca() : memref<8x8xi8>
  vector.transfer_write %T, %M[%c0, %c0] : vector<8x8xi8>, memref<8x8xi8>
  %MM = memref.cast %M : memref<8x8xi8> to memref<?x8xi8>

// CHECK-LABEL: Result(VL128):
// CHECK:( 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28 )
// CHECK:( 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48 )
  vector.print str "Result(VL128):\n"
  %c128 = arith.constant 128 : i32
  func.call @setArmVLBits(%c128) : (i32) -> ()
  func.call @transfer_read_scalable_non_trailing(%MM) : (memref<?x8xi8>) -> ()

// CHECK-LABEL: Result(VL256):
// CHECK: ( 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48 )
// CHECK: ( 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88 )
  vector.print str "Result(VL256):\n"
  %c256 = arith.constant 256 : i32
  func.call @setArmVLBits(%c256) : (i32) -> ()
  func.call @transfer_read_scalable_non_trailing(%MM) : (memref<?x8xi8>) -> ()

  return
}

func.func private @printVec(%v : vector<[32]xi8>) {
  %v0 = vector.scalable.extract %v[0] : vector<[16]xi8> from vector<[32]xi8>
  %v1 = vector.scalable.extract %v[16] : vector<[16]xi8> from vector<[32]xi8>
  vector.print %v0 : vector<[16]xi8>
  vector.print %v1 : vector<[16]xi8>
  return
}

func.func private @setArmVLBits(%bits : i32)
