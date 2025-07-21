// REQUIRES: arm-emulator

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --convert-vector-to-scf --convert-scf-to-cf  --convert-vector-to-llvm='enable-arm-neon enable-arm-i8mm' \
// DEFINE:   --expand-strided-metadata --convert-to-llvm --finalize-memref-to-llvm  \
// DEFINE:   --lower-affine --convert-arith-to-llvm --reconcile-unrealized-casts \
// DEFINE: -o %t

// DEFINE: %{entry_point} = main

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void  --march=aarch64 --mattr="+neon,+i8mm" \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} && FileCheck %s --input-file=%t -check-prefix CHECK-IR && %{run} | FileCheck %s

#packed_maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

//
// Test the lowering of `vector.contract` using the `LowerContractionToNeonI8MMPattern`
//
// The operation that the `vector.contract` in this test performs is matrix
// multiplication with accumulate
//     OUT = ACC + LHS * RHS
// of two 8-bit integer matrices LHS and RHS, and a 32-bit integer matrix ACC
// into a 32-bit integer matrix OUT. The LHS and RHS can be sign- or zero- extended,
// this test covers all the possible variants.
//
// Tested are calculations as well as that the relevant `ArmNeon` dialect
// operations ('arm_neon.smmla`, arm_neon.ummla`, etc) are emitted.
//
// That pattern above handles (therefore this test prepares) input/output vectors with
// specific shapes:
//   * LHS:      vector<MxKxi8>
//   * RHS:      vector<NxKxi8>
//   * ACC, OUT: vector<MxNxi32>
// where the M and N are even and K is divisible by 8.
// Note that the RHS is transposed.
// This data layout makes it efficient to load data into SIMD
// registers in the layout expected by FEAT_I8MM instructions.
// Such a `vector.contract` is representative of the code we aim to generate
// by vectorisation of `linalg.mmt4d`.
//
// In this specific test we use M == 4, N == 4, and K == 8.
//

// Test the operation where both LHS and RHS are interpreted as signed, hence
// we ultimately emit and execute the `smmla` instruction.

// CHECK-IR-LABEL: llvm.func @test_smmla
// CHECK-IR-COUNT-4: arm_neon.intr.smmla
func.func @test_smmla() {

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8

  // Accumulator test data
  %acc_cst = arith.constant dense<[[-44,  20,  44, -46],
                                   [ -8,  25, -34,  26],
                                   [-20, -36,  -3,  39],
                                   [-48, -31, -25, -21]]> : vector<4x4xi32>

  %acc_mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x4xi32>, vector<4x4xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[-35, -27, -36, -31,  23, -34,  -8, -33],
                                   [-20,  17, -32, -47,  37,  22,  -7, -21],
                                   [ -7, -35,  20,  -4,  39,  46, -23,  40],
                                   [ 40,  27,  37,  43,  38,  -6,  37,  49]]> : vector<4x8xi8>

  %lhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[-17, -50,  -1,  48, -13,  22,  39,  33],
                                   [-35, -24,  37, -32,  33,  30, -11, -17],
                                   [-28,  31,   3, -44, -15, -27,  22,  35],
                                   [-23,  39,  48,  26, -23,  32, -39, -38]]> : vector<4x8xi8>

  %rhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>


  // Matrix multiplication and accumulate with transposed RHS.
  %0 = arith.extsi %lhs : vector<4x8xi8> to vector<4x8xi32>
  %1 = arith.extsi %rhs : vector<4x8xi8> to vector<4x8xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x8xi32>, vector<4x8xi32> into vector<4x4xi32>

  // Display the result of the multiplication
  vector.print str "Result(SMMLA):\n"
  %u0 = vector.extract %2[0] : vector<4xi32> from vector<4x4xi32>
  %u1 = vector.extract %2[1] : vector<4xi32> from vector<4x4xi32>
  %u2 = vector.extract %2[2] : vector<4xi32> from vector<4x4xi32>
  %u3 = vector.extract %2[3] : vector<4xi32> from vector<4x4xi32>
  vector.print %u0 : vector<4xi32>
  vector.print %u1 : vector<4xi32>
  vector.print %u2 : vector<4xi32>
  vector.print %u3 : vector<4xi32>

  return
}

// Test the operation where both LHS and RHS are interpreted as unsigned, hence
// we ultimately emit and execute the `ummla` instruction.

// CHECK-IR-LABEL: llvm.func @test_ummla
// CHECK-IR-COUNT-4: arm_neon.intr.ummla
func.func @test_ummla() {

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8

  // Accumulator test data
  %acc_cst = arith.constant dense<[[16, 16, 48, 40],
                                   [40, 24, 35, 12],
                                   [33, 24, 29, 19],
                                   [28, 13, 33, 18]]> : vector<4x4xi32>

  %acc_mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x4xi32>, vector<4x4xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[35, 42, 37, 49, 36, 36, 23, 33],
                                   [39, 34, 33, 45, 43, 10, 44, 47],
                                   [18, 35, 29, 25, 36, 33, 28, 29],
                                   [26, 49, 43, 32, 27, 16, 45, 33]]> : vector<4x8xi8>

  %lhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[18, 31, 37, 35, 44, 22, 37, 28],
                                   [21, 22, 49, 39, 30, 28, 35, 37],
                                   [21, 47, 39, 35, 23, 43, 24, 49],
                                   [49, 49, 40, 32, 37, 20, 47, 40]]> : vector<4x8xi8>

  %rhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>

  // Matrix multiplication and accumulate with transposed RHS.
  %0 = arith.extui %lhs : vector<4x8xi8> to vector<4x8xi32>
  %1 = arith.extui %rhs : vector<4x8xi8> to vector<4x8xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x8xi32>, vector<4x8xi32> into vector<4x4xi32>

  // Display the result of the multiplication
  vector.print str "Result(UMMLA):\n"
  %u0 = vector.extract %2[0] : vector<4xi32> from vector<4x4xi32>
  %u1 = vector.extract %2[1] : vector<4xi32> from vector<4x4xi32>
  %u2 = vector.extract %2[2] : vector<4xi32> from vector<4x4xi32>
  %u3 = vector.extract %2[3] : vector<4xi32> from vector<4x4xi32>
  vector.print %u0 : vector<4xi32>
  vector.print %u1 : vector<4xi32>
  vector.print %u2 : vector<4xi32>
  vector.print %u3 : vector<4xi32>

  return
}

// Test the operation where LHS is interpreted as unsigned and RHS is
// interpreted as signed, hence we ultimately emit and execute the `usmmla`
// instruction.

// CHECK-IR-LABEL: llvm.func @test_usmmla
// CHECK-IR-COUNT-4: arm_neon.intr.usmmla
func.func @test_usmmla() {

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8

  // Accumulator test data
  %acc_cst = arith.constant dense<[[-44,  20,  44, -46],
                                   [ -8,  25, -34,  26],
                                   [-20, -36,  -3,  39],
                                   [-48, -31, -25, -21]]> : vector<4x4xi32>

  %acc_mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x4xi32>, vector<4x4xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[153, 161,  24, 157, 211, 154,  52,  27],
                                   [168,  77, 136, 124, 249,  28,  13, 122],
                                   [ 97,  82, 181,  39,  53,  25,  80, 240],
                                   [184, 227, 106, 165, 126, 113, 121, 228]]> : vector<4x8xi8>

  %lhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[ 40,  27,  37,  43,  38,  -6,  37,  49],
                                   [-17, -50,  -1,  48, -13,  22,  39,  33],
                                   [-35, -24,  37, -32,  33,  30, -11, -17],
                                   [-28,  31,   3, -44, -15, -27,  22,  35]]> : vector<4x8xi8>

  %rhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>

  // Matrix multiplication and accumulate with transposed RHS.
  %0 = arith.extui %lhs : vector<4x8xi8> to vector<4x8xi32>
  %1 = arith.extsi %rhs : vector<4x8xi8> to vector<4x8xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x8xi32>, vector<4x8xi32> into vector<4x4xi32>

  // Display the result of the multiplication
  vector.print str "Result(USMMLA):\n"
  %u0 = vector.extract %2[0] : vector<4xi32> from vector<4x4xi32>
  %u1 = vector.extract %2[1] : vector<4xi32> from vector<4x4xi32>
  %u2 = vector.extract %2[2] : vector<4xi32> from vector<4x4xi32>
  %u3 = vector.extract %2[3] : vector<4xi32> from vector<4x4xi32>
  vector.print %u0 : vector<4xi32>
  vector.print %u1 : vector<4xi32>
  vector.print %u2 : vector<4xi32>
  vector.print %u3 : vector<4xi32>

  return
}

// Test the operation where LHS is interpreted as signed and RHS is interpreted
// as unsigned. In this test we ultimately emit end execute the `usmmla`
// instruction with reversed operands, see `LowerContractionToNeonI8MMPattern.cpp`
// for more details.

// CHECK-IR-LABEL: llvm.func @test_summla
// CHECK-IR-COUNT-4: arm_neon.intr.usmmla
func.func @test_summla() {

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8

  // Accumulator test data
  %acc_cst = arith.constant dense<[[-44,  20,  44, -46],
                                   [ -8,  25, -34,  26],
                                   [-20, -36,  -3,  39],
                                   [-48, -31, -25, -21]]> : vector<4x4xi32>

  %acc_mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x4xi32>, vector<4x4xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[-35, -27, -36, -31,  23, -34,  -8, -33],
                                   [-20,  17, -32, -47,  37,  22,  -7, -21],
                                   [ -7, -35,  20,  -4,  39,  46, -23,  40],
                                   [ 40,  27,  37,  43,  38,  -6,  37,  49]]> : vector<4x8xi8>

  %lhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[125, 171, 138, 187, 108, 175,  82,  99],
                                   [221,  25, 164,  97, 156, 221, 218, 177],
                                   [171, 160, 219, 191, 144,  45, 161, 210],
                                   [223, 165, 123,  99, 108,  86,  37,  92]]> : vector<4x8xi8>

  %rhs_mem = memref.alloca() : memref<4x8xi8>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] : vector<4x8xi8>, memref<4x8xi8>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x8xi8>, vector<4x8xi8>

  // Matrix multiplication and accumulate with transposed RHS.
  %0 = arith.extsi %lhs : vector<4x8xi8> to vector<4x8xi32>
  %1 = arith.extui %rhs : vector<4x8xi8> to vector<4x8xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x8xi32>, vector<4x8xi32> into vector<4x4xi32>

  // Display the result of the multiplication
  vector.print str "Result(SUMMLA (i.e. USMMLA transposed)):\n"
  %u0 = vector.extract %2[0] : vector<4xi32> from vector<4x4xi32>
  %u1 = vector.extract %2[1] : vector<4xi32> from vector<4x4xi32>
  %u2 = vector.extract %2[2] : vector<4xi32> from vector<4x4xi32>
  %u3 = vector.extract %2[3] : vector<4xi32> from vector<4x4xi32>
  vector.print %u0 : vector<4xi32>
  vector.print %u1 : vector<4xi32>
  vector.print %u2 : vector<4xi32>
  vector.print %u3 : vector<4xi32>

  return
}

func.func @main() {
// CHECK-LABEL: Result(SMMLA):
// CHECK: ( -1999,  1941,   685, -2879 )
// CHECK: ( -3705,  2952,   987,  -685 )
// CHECK: (  2565,  4157, -1589,  -357 )
// CHECK: (  2383, -2252,    32, -1365 )
  func.call @test_smmla() : () -> ()

// CHECK-LABEL: Result(UMMLA):
// CHECK: ( 9183,  9513, 10460, 11314 )
// CHECK: ( 9648,  9812, 10092, 12088 )
// CHECK: ( 7548,  7625,  8398,  9044 )
// CHECK: ( 8855,  9046,  9685, 11191 )
  func.call @test_ummla() : () -> ()

// CHECK-LABEL: Result(USMMLA):
// CHECK: ( 28403,    445,  -2759, -11409 )
// CHECK: ( 34908,   1047,    142,  -7274 )
// CHECK: ( 31032,   6807,  -2378,   7382 )
// CHECK: ( 44217,   6396, -10930,    623 )
  func.call @test_usmmla() : () -> ()

// CHECK-LABEL: Result(SUMMLA (i.e. USMMLA transposed)):
// CHECK: ( -27190, -28812, -30502, -23575 )
// CHECK: (  -7613,  -8386, -15938,  -6521 )
// CHECK: (   9468,  18750,   9199,   5764 )
// CHECK: (  33655,  41064,  48900,  31627 )
  func.call @test_summla() : () -> ()

  return
}
