// REQUIRES: arm-emulator

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --convert-vector-to-scf --convert-scf-to-cf  --convert-vector-to-llvm='enable-arm-sve enable-arm-i8mm' \
// DEFINE:   --expand-strided-metadata --convert-to-llvm --finalize-memref-to-llvm  --reconcile-unrealized-casts \
// DEFINE: -o %t

// DEFINE: %{entry_point} = main

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void  --march=aarch64 --mattr="+sve,+i8mm" \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} && FileCheck %s --input-file=%t -check-prefix CHECK-IR && %{run} | FileCheck %s

#packed_maps = [
  affine_map<(d0, d1, d2) -> (d0, d2)>,
  affine_map<(d0, d1, d2) -> (d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1)>
]

func.func private @setArmVLBits(%bits : i32)

func.func private @prepareAccTestData(%in: vector<4x4xi32>) -> vector<4x[4]xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32

  %mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %in, %mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>

  %flat_mem = memref.collapse_shape %mem [[0, 1]] : memref<4x4xi32> into memref<16xi32>
  %flat_vec = vector.transfer_read %flat_mem[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32>, vector<[16]xi32>
  %out = vector.shape_cast %flat_vec : vector<[16]xi32> to vector<4x[4]xi32>

  return %out : vector<4x[4]xi32>
}

func.func private @prepareLHSTestData(%in: vector<4x8xi8>) -> vector<4x8xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %in, %mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>

  %out = vector.transfer_read %mem[%c0, %c0], %c0_i8 : memref<4x8xi8>, vector<4x8xi8>

  return %out :  vector<4x8xi8>
}

func.func private @prepareRHSTestData(%in: vector<4x8xi8>) -> vector<[32]xi8> {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8

  %mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %in, %mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>

  %flat_mem = memref.collapse_shape %mem [[0, 1]] : memref<4x8xi8> into memref<32xi8>
  %flat_vec = vector.transfer_read %flat_mem[%c0], %c0_i8 {in_bounds = [true]} : memref<32xi8>, vector<[32]xi8>

  return %flat_vec : vector<[32]xi8>
}

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

  %acc = func.call @prepareAccTestData(%acc_cst) : (vector<4x4xi32>) -> vector<4x[4]xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[-35, -27, -36, -31,  23, -34,  -8, -33],
                               [-20,  17, -32, -47,  37,  22,  -7, -21],
                               [ -7, -35,  20,  -4,  39,  46, -23,  40],
                               [ 40,  27,  37,  43,  38,  -6,  37,  49]]> : vector<4x8xi8>

  %lhs = func.call @prepareLHSTestData(%lhs_cst) : (vector<4x8xi8>) -> vector<4x8xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[-17, -50,  -1,  48, -13,  22,  39,  33],
                                   [-35, -24,  37, -32,  33,  30, -11, -17],
                                   [-28,  31,   3, -44, -15, -27,  22,  35],
                                   [-23,  39,  48,  26, -23,  32, -39, -38]]> : vector<4x8xi8>
  %rhs_flat = func.call @prepareRHSTestData(%rhs_cst) : (vector<4x8xi8>) -> vector<[32]xi8>
  %rhs = vector.shape_cast %rhs_flat : vector<[32]xi8> to vector<[4]x8xi8>

// CHECK-IR-COUNT-4: arm_sve.intr.smmla

  // Matrix multiplication
  %0 = arith.extsi %lhs : vector<4x8xi8> to vector<4x8xi32>
  %1 = arith.extsi %rhs : vector<[4]x8xi8> to vector<[4]x8xi32>
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

// CHECK: ( -1999,  1941,   685, -2879 )
// CHECK: ( -3705,  2952,   987,  -685 )
// CHECK: (  2565,  4157, -1589,  -357 )
// CHECK: (  2383, -2252,    32, -1365 )
  return
}
