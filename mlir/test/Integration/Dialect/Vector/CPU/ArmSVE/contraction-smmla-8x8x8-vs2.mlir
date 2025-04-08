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
  %c256 = arith.constant 256 : i32
  func.call @setArmVLBits(%c256) : (i32) -> ()

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8


  // Accumulator test data
  %acc_cst = arith.constant dense<[[-44,  20,  44, -46,  -8,  25, -34,  26],
                                   [-20, -36,  -3,  39, -48, -31, -25, -21],
                                   [-35, -27, -36, -31,  23, -34,  -8, -33],
                                   [-20,  17, -32, -47,  37,  22,  -7, -21],
                                   [ -7, -35,  20,  -4,  39,  46, -23,  40],
                                   [ 40,  27,  37,  43,  38,  -6,  37,  49],
                                   [-17, -50,  -1,  48, -13,  22,  39,  33],
                                   [-35, -24,  37, -32,  33,  30, -11, -17]]> : vector<8x8xi32>
  %acc_m = memref.alloca() : memref<8x8xi32>
  vector.transfer_write %acc_cst, %acc_m[%c0, %c0] : vector<8x8xi32>, memref<8x8xi32>

  %acc_m1 = memref.collapse_shape %acc_m [[0, 1]] : memref<8x8xi32> into memref<64xi32>
  %acc_flat = vector.transfer_read %acc_m1[%c0], %c0_i32 {in_bounds = [true]} : memref<64xi32>, vector<[32]xi32>
  %acc = vector.shape_cast %acc_flat : vector<[32]xi32> to vector<8x[4]xi32>

  vector.print str "ACC:\n"
  %acc0 = vector.extract %acc[0] : vector<[4]xi32> from vector<8x[4]xi32>
  %acc1 = vector.extract %acc[1] : vector<[4]xi32> from vector<8x[4]xi32>
  %acc2 = vector.extract %acc[2] : vector<[4]xi32> from vector<8x[4]xi32>
  %acc3 = vector.extract %acc[3] : vector<[4]xi32> from vector<8x[4]xi32>
  %acc4 = vector.extract %acc[4] : vector<[4]xi32> from vector<8x[4]xi32>
  %acc5 = vector.extract %acc[5] : vector<[4]xi32> from vector<8x[4]xi32>
  %acc6 = vector.extract %acc[6] : vector<[4]xi32> from vector<8x[4]xi32>
  %acc7 = vector.extract %acc[7] : vector<[4]xi32> from vector<8x[4]xi32>
  vector.print %acc0 : vector<[4]xi32>
  vector.print %acc1 : vector<[4]xi32>
  vector.print %acc2 : vector<[4]xi32>
  vector.print %acc3 : vector<[4]xi32>
  vector.print %acc4 : vector<[4]xi32>
  vector.print %acc5 : vector<[4]xi32>
  vector.print %acc6 : vector<[4]xi32>
  vector.print %acc7 : vector<[4]xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[-28,  31,   3, -44, -15, -27,  22,  35],
                                   [-23,  39,  48,  26, -23,  32, -39, -38],
                                   [ -3,   9,  43, -30, -32,  39,  41, -39],
                                   [-13, -21, -25,  27,  47, -36, -11, -11],
                                   [ -4, -20,  36,  11,  13, -23,  24, -13],
                                   [-20,  30,  -5,   1,  42, -37, -22,  35],
                                   [-22,  38,  -4,  44,  25, -31,  23, -39],
                                   [-45,  -4, -31, -24,  14, -41, -47,  22]]> : vector<8x8xi8>

  %lhs_m = memref.alloca() : memref<8x8xi8>
  vector.transfer_write %lhs_cst, %lhs_m[%c0, %c0] : vector<8x8xi8>, memref<8x8xi8>
  %lhs = vector.transfer_read %lhs_m[%c0, %c0], %c0_i8 : memref<8x8xi8>, vector<8x8xi8>

  vector.print str "LHS:\n"
  %lhs0 = vector.extract %lhs[0] : vector<8xi8> from vector<8x8xi8>
  %lhs1 = vector.extract %lhs[1] : vector<8xi8> from vector<8x8xi8>
  %lhs2 = vector.extract %lhs[2] : vector<8xi8> from vector<8x8xi8>
  %lhs3 = vector.extract %lhs[3] : vector<8xi8> from vector<8x8xi8>
  %lhs4 = vector.extract %lhs[4] : vector<8xi8> from vector<8x8xi8>
  %lhs5 = vector.extract %lhs[5] : vector<8xi8> from vector<8x8xi8>
  %lhs6 = vector.extract %lhs[6] : vector<8xi8> from vector<8x8xi8>
  %lhs7 = vector.extract %lhs[7] : vector<8xi8> from vector<8x8xi8>
  vector.print %lhs0 : vector<8xi8>
  vector.print %lhs1 : vector<8xi8>
  vector.print %lhs2 : vector<8xi8>
  vector.print %lhs3 : vector<8xi8>
  vector.print %lhs4 : vector<8xi8>
  vector.print %lhs5 : vector<8xi8>
  vector.print %lhs6 : vector<8xi8>
  vector.print %lhs7 : vector<8xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[-40, -11, -36,  36,  -1,  20,  14, -32],
                                   [ 46, -45, -48, -46, -24,  31, -36,  22],
                                   [  2,  36,  45, -29, -37, -49, -20, -35],
                                   [ -6,  23,  23,  15,  20,   4,  -8,  -2],
                                   [-35,  -6,  16,  49, -50,   9, -44,  13],
                                   [ 24,   1,  -4, -44,  41,  15, -43,  44],
                                   [ 44,   0, -10,  41,  22,  44, -40,   0],
                                   [-33,  19,  27,  22,  38, -17,  23,  -9]]> : vector<8x8xi8>

  %rhs_m = memref.alloca() : memref<8x8xi8>
  vector.transfer_write %rhs_cst, %rhs_m[%c0, %c0] : vector<8x8xi8>, memref<8x8xi8>

  %rhs_m1 = memref.collapse_shape %rhs_m [[0, 1]] : memref<8x8xi8> into memref<64xi8>
  %rhs_flat = vector.transfer_read %rhs_m1[%c0], %c0_i8 {in_bounds = [true]} : memref<64xi8>, vector<[32]xi8>

  vector.print str "RHS:\n"
  %rhs0 = vector.scalable.extract %rhs_flat[ 0] : vector<[16]xi8> from vector<[32]xi8>
  %rhs1 = vector.scalable.extract %rhs_flat[16] : vector<[16]xi8> from vector<[32]xi8>
  vector.print %rhs0 : vector<[16]xi8>
  vector.print %rhs1 : vector<[16]xi8>

  %rhs = vector.shape_cast %rhs_flat : vector<[32]xi8> to vector<[4]x8xi8>

  // Matrix multiplication
  %0 = arith.extsi %lhs : vector<8x8xi8> to vector<8x8xi32>
  %1 = arith.extsi %rhs : vector<[4]x8xi8> to vector<[4]x8xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<8x8xi32>, vector<[4]x8xi32> into vector<8x[4]xi32>

  // Display the result of the multilication
  vector.print str "Result:\n"
  %u0 = vector.extract %2[0] : vector<[4]xi32> from vector<8x[4]xi32>
  %u1 = vector.extract %2[1] : vector<[4]xi32> from vector<8x[4]xi32>
  %u2 = vector.extract %2[2] : vector<[4]xi32> from vector<8x[4]xi32>
  %u3 = vector.extract %2[3] : vector<[4]xi32> from vector<8x[4]xi32>
  %u4 = vector.extract %2[4] : vector<[4]xi32> from vector<8x[4]xi32>
  %u5 = vector.extract %2[5] : vector<[4]xi32> from vector<8x[4]xi32>
  %u6 = vector.extract %2[6] : vector<[4]xi32> from vector<8x[4]xi32>
  %u7 = vector.extract %2[7] : vector<[4]xi32> from vector<8x[4]xi32>
  vector.print %u0 : vector<[4]xi32>
  vector.print %u1 : vector<[4]xi32>
  vector.print %u2 : vector<[4]xi32>
  vector.print %u3 : vector<[4]xi32>
  vector.print %u4 : vector<[4]xi32>
  vector.print %u5 : vector<[4]xi32>
  vector.print %u6 : vector<[4]xi32>
  vector.print %u7 : vector<[4]xi32>


// CHECK: ( -2294, -1282,  2728,  -410, -1328,   882, -5498,   732 )
// CHECK: (  1012, -4237,  4154,  2624,  5225, -2338,  2011,  1374 )
// CHECK: (    -8, -1611,  2905,    -1, -1068, -3155, -2428,   153 )
// CHECK: (  2034, -1768, -2092,   284,  -792,   -23,   668,  2172 )
// CHECK: (  -248, -3728,  1214,   555,  -668, -2114, -1794,  2560 )
// CHECK: ( -1484, -2642,   297,  1551,  -483,  3173,  -576,  2570 )
// CHECK: (  3098, -7851,  1366,  1892,  -427, -4533,  -819,  4698 )
// CHECK: (  -135,  1247,   765,  -479,  1245,  3074, -2281,   -23 )
  return
}
