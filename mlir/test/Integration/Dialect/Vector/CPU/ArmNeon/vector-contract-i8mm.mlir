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
// In this specific test we use M == 4, N == 4, and K == 16.
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
  %acc_cst = arith.constant dense<[[ -1,  -9,  -4,   0],
                                   [  6,   5,   7,   2],
                                   [ -8,  -7,   9, -10],
                                   [  9,   4,  -4,   0]]> : vector<4x4xi32>

  %acc_mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x4xi32>, vector<4x4xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[ -4,  -4,  -4,  -6,   0,   1,   6,   2,  -1,   4,   5,  -8,   9,   5,   4,   9],
                                   [ -1,   6,   0,   7,  -7,   8,   5,   8,  -7,   6,  -2,   1,   1,   5,  -4,  -4],
                                   [  4, -10,  10,  -3,   5,   3,   2,   3,  -7,   9,  -9, -10,   7,  -8,  -5,  -2],
                                   [  9,   5,   8,   9,   6,  -3,  -9,   7,  -4,  -7,  -2,   7,  -8,   2,   8,   7]]> : vector<4x16xi8>

  %lhs_mem = memref.alloca() : memref<4x16xi8>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] : vector<4x16xi8>, memref<4x16xi8>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x16xi8>, vector<4x16xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[  1,   2,  -3,   5,  10,   8,  10,  -2,   1,  10,  -5,   2,   4,   3,  -9,   4],
                                   [ -3,  -3,  -3,   4,   6,  -1,   0,  -5,   6,   3,  -1,   9,  -3,   3,  -2,   4],
                                   [  1,   9,  -1,   1,  -5,   4,   9, -10,  -1,  -7,  10,  -2,   0,  -3,   4,   7],
                                   [ -4, -10,   8, -10,  -5,  -8,  -6,   7,   4,  -2,  10,   3,  -9,   5,   2,  -1]]> : vector<4x16xi8>

  %rhs_mem = memref.alloca() : memref<4x16xi8>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] : vector<4x16xi8>, memref<4x16xi8>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x16xi8>, vector<4x16xi8>


  // Matrix multiplication and accumulate with transposed RHS.
  %0 = arith.extsi %lhs : vector<4x16xi8> to vector<4x16xi32>
  %1 = arith.extsi %rhs : vector<4x16xi8> to vector<4x16xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x16xi32>, vector<4x16xi32> into vector<4x4xi32>

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
  %acc_cst = arith.constant dense<[[39, 39, 46, 30],
                                   [22, 48, 61, 54],
                                   [41, 63, 27, 10],
                                   [37, 30, 16, 45]]> : vector<4x4xi32>

  %acc_mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x4xi32>, vector<4x4xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[ 6,  6, 38, 30, 60,  4, 42, 11, 16, 12, 30, 41, 14, 55, 47, 25],
                                   [ 2, 19, 25, 29, 15, 23, 14, 19,  9, 16, 42, 17, 58, 62, 30,  3],
                                   [62, 50, 47, 18,  3, 48, 23,  8, 43, 29, 43, 15,  6, 38, 46, 25],
                                   [32, 27, 52, 39, 47, 26, 26, 13, 23, 29, 24, 44, 23, 45, 35, 51]]> : vector<4x16xi8>

  %lhs_mem = memref.alloca() : memref<4x16xi8>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] : vector<4x16xi8>, memref<4x16xi8>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x16xi8>, vector<4x16xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[33,  0, 49, 34, 37,  8, 25, 19, 15, 26, 23, 18, 19, 16, 39, 33],
                                   [22, 17, 53, 58,  6, 35, 54, 23,  8, 53, 21, 27, 49, 25, 34, 12],
                                   [27, 18, 53, 53, 49, 11, 12, 39, 62, 47, 59, 29, 20, 18, 52, 25],
                                   [27, 40, 11, 52, 37, 60, 29, 44, 46, 25, 13, 33, 14, 53, 56, 39]]> : vector<4x16xi8>

  %rhs_mem = memref.alloca() : memref<4x16xi8>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] : vector<4x16xi8>, memref<4x16xi8>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x16xi8>, vector<4x16xi8>

  // Matrix multiplication and accumulate with transposed RHS.
  %0 = arith.extui %lhs : vector<4x16xi8> to vector<4x16xi32>
  %1 = arith.extui %rhs : vector<4x16xi8> to vector<4x16xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x16xi32>, vector<4x16xi32> into vector<4x4xi32>

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
  %acc_cst = arith.constant dense<[[-50,  22, -15,   6],
                                   [  0, -46,  32, -59],
                                   [-62, -60, -38,  17],
                                   [-50,   8, -12,  22]]> : vector<4x4xi32>

  %acc_mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x4xi32>, vector<4x4xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[ 6,  6, 38, 30, 60,  4, 42, 11, 16, 12, 30, 41, 14, 55, 47, 25],
                                   [ 2, 19, 25, 29, 15, 23, 14, 19,  9, 16, 42, 17, 58, 62, 30,  3],
                                   [62, 50, 47, 18,  3, 48, 23,  8, 43, 29, 43, 15,  6, 38, 46, 25],
                                   [32, 27, 52, 39, 47, 26, 26, 13, 23, 29, 24, 44, 23, 45, 35, 51]]> : vector<4x16xi8>

  %lhs_mem = memref.alloca() : memref<4x16xi8>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] : vector<4x16xi8>, memref<4x16xi8>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x16xi8>, vector<4x16xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[ -9, -10,   7,  -8,  -5,  -2,   9,   5,   8,   9,   6,  -3,  -9,   7,  -4,  -7],
                                   [ -2,   7,  -8,   2,   8,   7,   1,   2,  -3,   5,   8,  -2,   1,  -5,   2,   4],
                                   [  3,  -9,   4,  -3,  -3,  -3,   4,   6,  -1,   0,  -5,   6,   3,  -1,   9,  -3],
                                   [  3,  -2,   4,   1,   9,  -1,   1,  -5,   4,   9, -10,  -1,  -7,  -2,   0,  -3]]> : vector<4x16xi8>

  %rhs_mem = memref.alloca() : memref<4x16xi8>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] : vector<4x16xi8>, memref<4x16xi8>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x16xi8>, vector<4x16xi8>

  // Matrix multiplication and accumulate with transposed RHS.
  %0 = arith.extui %lhs : vector<4x16xi8> to vector<4x16xi32>
  %1 = arith.extsi %rhs : vector<4x16xi8> to vector<4x16xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x16xi32>, vector<4x16xi32> into vector<4x4xi32>

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
  %acc_cst = arith.constant dense<[[-61,  52,   8, -54],
                                   [-25, -50,  22, -15],
                                   [  6,   0, -46,  32],
                                   [-59, -62, -60, -38]]> : vector<4x4xi32>

  %acc_mem = memref.alloca() : memref<4x4xi32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] : vector<4x4xi32>, memref<4x4xi32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x4xi32>, vector<4x4xi32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[ -4,  -4,  -4,  -6,   0,   1,   6,   2,  -1,   4,   5,  -8,   9,   5,   4,   9],
                                   [ -1,   6,   0,   7,  -7,   8,   5,   8,  -7,   6,  -2,   1,   1,   5,  -4,  -4],
                                   [  4, -10,  -3,   5,   3,   2,   3,  -7,   9,  -9, -10,   7,  -8,  -5,  -2,   9],
                                   [  5,   8,   9,   6,  -3,  -9,   7,  -4,  -7,  -2,   7,  -8,   2,   8,   7,   1]]> : vector<4x16xi8>

  %lhs_mem = memref.alloca() : memref<4x16xi8>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] : vector<4x16xi8>, memref<4x16xi8>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x16xi8>, vector<4x16xi8>

  // RHS test data
  %rhs_cst = arith.constant dense<[[12, 39, 62, 47, 59, 29, 20, 18, 52, 25, 27, 40, 11, 52, 37, 60],
                                   [29, 44, 46, 25, 13, 33, 14, 53, 56, 39, 39, 39, 46, 30, 22, 48],
                                   [61, 54, 41, 63, 27, 10, 37, 30, 16, 45, 41, 51, 39, 28, 13, 28],
                                   [21, 28, 24, 40, 46, 30, 11, 19,  9, 11,  5, 46, 19, 26,  0,  9]]> : vector<4x16xi8>

  %rhs_mem = memref.alloca() : memref<4x16xi8>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] : vector<4x16xi8>, memref<4x16xi8>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4x16xi8>, vector<4x16xi8>

  // Matrix multiplication and accumulate with transposed RHS.
  %0 = arith.extsi %lhs : vector<4x16xi8> to vector<4x16xi32>
  %1 = arith.extui %rhs : vector<4x16xi8> to vector<4x16xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x16xi32>, vector<4x16xi32> into vector<4x4xi32>

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
// CHECK: (   82,  -63,   95,   11 )
// CHECK: (  184,  -81,  -17, -172 )
// CHECK: (  168, -158, -251, -133 )
// CHECK: ( -139,   40,  -48,   75 )
  func.call @test_smmla() : () -> ()

// CHECK-LABEL: Result(UMMLA):
// CHECK: ( 12414, 13508, 16691, 16069 )
// CHECK: (  8935, 13219, 13408, 13644 )
// CHECK: ( 12223, 15233, 18131, 18553 )
// CHECK: ( 14459, 16573, 19443, 19417 )
  func.call @test_ummla() : () -> ()

// CHECK-LABEL: Result(USMMLA):
// CHECK: (  176,  483,  468,  265 )
// CHECK: (   23,  449,  192, -727 )
// CHECK: ( -128,  563,  -30,   66 )
// CHECK: ( -476,  657,  202,  334 )
  func.call @test_usmmla() : () -> ()

// CHECK-LABEL: Result(SUMMLA (i.e. USMMLA transposed)):
// CHECK: ( 300,  716,   54, -378 )
// CHECK: ( 244,  746, 1184,  689 )
// CHECK: ( 253, -655, -688,  115 )
// CHECK: ( 995,  574, 1490,  177 )
  func.call @test_summla() : () -> ()

  return
}
