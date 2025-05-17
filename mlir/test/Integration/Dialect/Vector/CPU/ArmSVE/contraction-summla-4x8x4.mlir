// REQUIRES: arm-emulator

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --convert-vector-to-scf --convert-scf-to-cf  --convert-vector-to-llvm='enable-arm-sve enable-arm-i8mm' \
// DEFINE:   --expand-strided-metadata --convert-to-llvm --finalize-memref-to-llvm  --reconcile-unrealized-casts \
// DEFINE: -o %t

// DEFINE: %{entry_point} = main

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void  --march=aarch64 --mattr="+sve,+i8mm" \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

#packed_maps = [
  affine_map<(d0, d1, d2) -> (d0, d2)>,
  affine_map<(d0, d1, d2) -> (d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1)>
]

func.func private @setArmVLBits(%bits : i32)

func.func @main() {
  %c128 = arith.constant 128 : i32
  func.call @setArmVLBits(%c128) : (i32) -> ()

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8

// Accumulator test data
  %acc_cst = arith.constant dense<[[-44,  20,  44, -46],
                                   [ -8,  25, -34,  26],
                                   [-20, -36,  -3,  39],
                                   [-48, -31, -25, -21]]> : vector<4x4xi32>
  %acc_m = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_m[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>

  %acc_m1 = memref.collapse_shape %acc_m [[0, 1]] : memref<4x4xi32> into memref<16xi32>
  %acc_flat = vector.transfer_read %acc_m1[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32>, vector<[16]xi32>
  %acc = vector.shape_cast %acc_flat : vector<[16]xi32> to vector<4x[4]xi32>

  vector.print str "ACC:\n"
  %acc0 = vector.extract %acc[0] : vector<[4]xi32> from vector<4x[4]xi32>
  %acc1 = vector.extract %acc[1] : vector<[4]xi32> from vector<4x[4]xi32>
  %acc2 = vector.extract %acc[2] : vector<[4]xi32> from vector<4x[4]xi32>
  %acc3 = vector.extract %acc[3] : vector<[4]xi32> from vector<4x[4]xi32>
  vector.print %acc0 : vector<[4]xi32>
  vector.print %acc1 : vector<[4]xi32>
  vector.print %acc2 : vector<[4]xi32>
  vector.print %acc3 : vector<[4]xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[-35, -27, -36, -31,  23, -34,  -8, -33],
                                   [-20,  17, -32, -47,  37,  22,  -7, -21],
                                   [ -7, -35,  20,  -4,  39,  46, -23,  40],
                                   [ 40,  27,  37,  43,  38,  -6,  37,  49]]> : vector<4x8xi8>

  %lhs_m = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %lhs_cst, %lhs_m[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %lhs = vector.transfer_read %lhs_m[%c0, %c0], %c0_i8 : memref<4x8xi8>, vector<4x8xi8>

  vector.print str "LHS:\n"
  %lhs0 = vector.extract %lhs[0] : vector<8xi8> from vector<4x8xi8>
  %lhs1 = vector.extract %lhs[1] : vector<8xi8> from vector<4x8xi8>
  %lhs2 = vector.extract %lhs[2] : vector<8xi8> from vector<4x8xi8>
  %lhs3 = vector.extract %lhs[3] : vector<8xi8> from vector<4x8xi8>
  vector.print %lhs0 : vector<8xi8>
  vector.print %lhs1 : vector<8xi8>
  vector.print %lhs2 : vector<8xi8>
  vector.print %lhs3 : vector<8xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[125, 171, 138, 187, 108, 175,  82,  99],
                                   [221,  25, 164,  97, 156, 221, 218, 177],
                                   [171, 160, 219, 191, 144,  45, 161, 210],
                                   [223, 165, 123,  99, 108,  86,  37,  92]]> : vector<4x8xi8>

  %rhs_m = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %rhs_cst, %rhs_m[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>

  %rhs_m1 = memref.collapse_shape %rhs_m [[0, 1]] : memref<4x8xi8> into memref<32xi8>
  %rhs_flat = vector.transfer_read %rhs_m1[%c0], %c0_i8 {in_bounds = [true]} : memref<32xi8>, vector<[32]xi8>

  vector.print str "RHS:\n"
  %rhs0 = vector.scalable.extract %rhs_flat[0] : vector<[16]xi8> from vector<[32]xi8>
  %rhs1 = vector.scalable.extract %rhs_flat[16] : vector<[16]xi8> from vector<[32]xi8>
  vector.print %rhs0 : vector<[16]xi8>
  vector.print %rhs1 : vector<[16]xi8>

  %rhs = vector.shape_cast %rhs_flat : vector<[32]xi8> to vector<[4]x8xi8>

  // Matrix multiplication
  %0 = arith.extsi %lhs : vector<4x8xi8> to vector<4x8xi32>
  %1 = arith.extui %rhs : vector<[4]x8xi8> to vector<[4]x8xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x8xi32>, vector<[4]x8xi32> into vector<4x[4]xi32>

  // Display the result of the multiplication
  vector.print str "Result:\n"
  %u0 = vector.extract %2[0] : vector<[4]xi32> from vector<4x[4]xi32>
  %u1 = vector.extract %2[1] : vector<[4]xi32> from vector<4x[4]xi32>
  %u2 = vector.extract %2[2] : vector<[4]xi32> from vector<4x[4]xi32>
  %u3 = vector.extract %2[3] : vector<[4]xi32> from vector<4x[4]xi32>
  vector.print %u0 : vector<[4]xi32>
  vector.print %u1 : vector<[4]xi32>
  vector.print %u2 : vector<[4]xi32>
  vector.print %u3 : vector<[4]xi32>

// CHECK: ( -27190, -28812, -30502, -23575 )
// CHECK: (  -7613,  -8386, -15938,  -6521 )
// CHECK: (   9468,  18750,   9199,   5764 )
// CHECK: (  33655,  41064,  48900,  31627 )
  return
}

