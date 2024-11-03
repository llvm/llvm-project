// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm="enable-x86vector" -convert-func-to-llvm -reconcile-unrealized-casts  | \
// RUN: mlir-translate  --mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="avx512bw,avx512vp2intersect" --dlopen=%mlir_c_runner_utils | \
// RUN: FileCheck %s

// Note: To run this test, your CPU must support AVX512 vp2intersect.

func.func @entry() -> i32 {
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1: i32
  %i2 = arith.constant 2: i32
  %i3 = arith.constant 7: i32
  %i4 = arith.constant 12: i32
  %i5 = arith.constant -10: i32
  %i6 = arith.constant -219: i32

  %v0 = vector.broadcast %i1 : i32 to vector<16xi32>
  %v1 = vector.insert %i2, %v0[1] : i32 into vector<16xi32>
  %v2 = vector.insert %i3, %v1[4] : i32 into vector<16xi32>
  %v3 = vector.insert %i4, %v2[6] : i32 into vector<16xi32>
  %v4 = vector.insert %i5, %v3[7] : i32 into vector<16xi32>
  %v5 = vector.insert %i0, %v4[10] : i32 into vector<16xi32>
  %v6 = vector.insert %i0, %v5[12] : i32 into vector<16xi32>
  %v7 = vector.insert %i3, %v6[13] : i32 into vector<16xi32>
  %v8 = vector.insert %i3, %v7[14] : i32 into vector<16xi32>
  %v9 = vector.insert %i0, %v8[15] : i32 into vector<16xi32>
  vector.print %v9 : vector<16xi32>
  // CHECK: ( 1, 2, 1, 1, 7, 1, 12, -10, 1, 1, 0, 1, 0, 7, 7, 0 )

  %w0 = vector.broadcast %i1 : i32 to vector<16xi32>
  %w1 = vector.insert %i2, %w0[4] : i32 into vector<16xi32>
  %w2 = vector.insert %i6, %w1[7] : i32 into vector<16xi32>
  %w3 = vector.insert %i4, %w2[8] : i32 into vector<16xi32>
  %w4 = vector.insert %i4, %w3[9] : i32 into vector<16xi32>
  %w5 = vector.insert %i4, %w4[10] : i32 into vector<16xi32>
  %w6 = vector.insert %i0, %w5[11] : i32 into vector<16xi32>
  %w7 = vector.insert %i0, %w6[12] : i32 into vector<16xi32>
  %w8 = vector.insert %i0, %w7[13] : i32 into vector<16xi32>
  %w9 = vector.insert %i0, %w8[15] : i32 into vector<16xi32>
  vector.print %w9 : vector<16xi32>
  // CHECK: ( 1, 1, 1, 1, 2, 1, 1, -219, 12, 12, 12, 0, 0, 0, 1, 0 )

  %k1, %k2 = x86vector.avx512.vp2intersect %v9, %w9 : vector<16xi32>

  vector.print %k1 : vector<16xi1>
  // CHECK: ( 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1 )

  vector.print %k2 : vector<16xi1>
  // CHECK: ( 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1 )

  return %i0 : i32
}
