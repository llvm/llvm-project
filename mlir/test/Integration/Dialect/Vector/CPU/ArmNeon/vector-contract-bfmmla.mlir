// REQUIRES: arm-emulator

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --convert-vector-to-scf --convert-scf-to-cf  --convert-vector-to-llvm='enable-arm-neon enable-arm-bf16' \
// DEFINE:   --expand-strided-metadata --convert-to-llvm --finalize-memref-to-llvm  \
// DEFINE:   --lower-affine --convert-arith-to-llvm --reconcile-unrealized-casts \
// DEFINE: -o %t

// DEFINE: %{entry_point} = main

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void  --march=aarch64 --mattr="+bf16" \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} && FileCheck %s --input-file=%t -check-prefix CHECK-IR && %{run} | FileCheck %s

#packed_maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

//
// Test the lowering of `vector.contract` using the `LowerContractionToNeonBFMMLAPattern`
//
// The operation that the `vector.contract` in this test performs is matrix
// multiplication with accumulate
//     OUT = ACC + LHS * RHS
// of two BFloat16 matrices LHS and RHS, and a Float32 matrix ACC into a Float32 OUT.
//
// Tested are calculations as well as that the relevant `ArmNeon` dialect
// operation (`arm_neon.intr.bfmmla`) is emitted.
//
// That pattern above handles (therefore this test prepares) input/output vectors with
// specific shapes:
//   * LHS:      vector<MxKxbf16>
//   * RHS:      vector<NxKxbf16>
//   * ACC, OUT: vector<MxNxf32>
// where the M and N are even and K is divisible by 4.
// Note that the RHS is transposed.
// This data layout makes it efficient to load data into SIMD
// registers in the layout expected by BFMMLA instruction.
// Such a `vector.contract` is representative of the code we aim to generate
// by vectorisation of `linalg.mmt4d`.
//
// In this specific test we use M == 4, N == 4, and K == 4.

// Note: In this and in the following test the seemingly unnecessary
// writes of test vectors to memory are done in order to introduce memory
// load operations on the path leading up to `vector.contract` since
// that's more representative of the typical usage when multiplying
// big, non-constant tensors.

// CHECK-IR-LABEL: llvm.func @matrix_by_matrix_mul_and_acc
// CHECK-IR-COUNT-4: arm_neon.intr.bfmmla
func.func @matrix_by_matrix_mul_and_acc() {

  %c0 = arith.constant 0 : index
  %c0_f32 = arith.constant 0.0 : f32
  %c0_bf16 = arith.constant 0.0 : bf16

  // Accumulator test data
  %acc_cst = arith.constant dense<[[ 0.7,  1.0, -0.1,  1.8],
                                   [-0.5,  0.9,  0.7, -0.7],
                                   [ 0.5, -1.3, -2.2,  0.1],
                                   [-0.7,  1.0,  1.7, -1.0]]> : vector<4x4xf32>

  %acc_mem = memref.alloca() : memref<4x4xf32>
  vector.transfer_write %acc_cst, %acc_mem[%c0, %c0] {in_bounds = [true, true] } : vector<4x4xf32>, memref<4x4xf32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_f32 {in_bounds = [true, true]} : memref<4x4xf32>, vector<4x4xf32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[ 0.1,  0.7, -0.9,  1.3],
                                   [-1.6,  0.7, -0.3, -0.3],
                                   [-0.4,  0.6,  0.8, -0.5],
                                   [-0.6, -1.0, -1.0, -1.0]]> : vector<4x4xbf16>

  %lhs_mem = memref.alloca() : memref<4x4xbf16>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0, %c0] {in_bounds = [true, true] } : vector<4x4xbf16>, memref<4x4xbf16>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_bf16 {in_bounds = [true, true]} : memref<4x4xbf16>, vector<4x4xbf16>

  // RHS test data
  %rhs_cst = arith.constant dense<[[ 0.6,  1.3,  0.1, -0.9],
                                   [ 0.5,  1.6,  1.8,  1.6],
                                   [-0.2,  0.4,  1.0,  0.4],
                                   [-1.3, -0.2, -2.2,  0.3]]> : vector<4x4xbf16>

  %rhs_mem = memref.alloca() : memref<4x4xbf16>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] {in_bounds = [true, true] } : vector<4x4xbf16>, memref<4x4xbf16>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_bf16 {in_bounds = [true, true]} : memref<4x4xbf16>, vector<4x4xbf16>

  // Matrix multiplication and accumulate with transposed RHS.
  %0 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<4x4xbf16>, vector<4x4xbf16> into vector<4x4xf32>

  // Display the result of the multiplication
  vector.print str "Result(BFMMLA):\n"
  %u0 = vector.extract %0[0] : vector<4xf32> from vector<4x4xf32>
  %u1 = vector.extract %0[1] : vector<4xf32> from vector<4x4xf32>
  %u2 = vector.extract %0[2] : vector<4xf32> from vector<4x4xf32>
  %u3 = vector.extract %0[3] : vector<4xf32> from vector<4x4xf32>
  vector.print %u0 : vector<4xf32>
  vector.print %u1 : vector<4xf32>
  vector.print %u2 : vector<4xf32>
  vector.print %u3 : vector<4xf32>

  return
}

// Test when the LHS is a one-dimensional vector.
// 
// In the vector by matrix case the dhapes ae as follows:
//   * LHS:      vector<Kxbf16>
//   * RHS:      vector<NxKxbf16>
//   * ACC, OUT: vector<Nxf32>
// N is even and K is divisible by 4.
// In this specific test we use N == 4, and K == 4.

// CHECK-IR-LABEL: llvm.func @vector_by_matrix_mul_and_acc
// CHECK-IR-COUNT-2: arm_neon.intr.bfmmla
func.func @vector_by_matrix_mul_and_acc() {
  %c0 = arith.constant 0 : index
  %c0_f32 = arith.constant 0.0 : f32
  %c0_bf16 = arith.constant 0.0 : bf16

  // Accumulator test data
  %acc_cst = arith.constant dense<[0.7,  1.0, -0.1,  1.8]> : vector<4xf32>

  %acc_mem = memref.alloca() : memref<4xf32>
  vector.transfer_write %acc_cst, %acc_mem[%c0] {in_bounds = [true] } : vector<4xf32>, memref<4xf32>
  %acc = vector.transfer_read %acc_mem[%c0], %c0_f32 {in_bounds = [true]} : memref<4xf32>, vector<4xf32>

  // LHS test data
  %lhs_cst = arith.constant dense<[0.1,  0.7, -0.9,  1.3]> : vector<4xbf16>

  %lhs_mem = memref.alloca() : memref<4xbf16>
  vector.transfer_write %lhs_cst, %lhs_mem[%c0] {in_bounds = [true] } : vector<4xbf16>, memref<4xbf16>
  %lhs = vector.transfer_read %lhs_mem[%c0], %c0_bf16 {in_bounds = [true]} : memref<4xbf16>, vector<4xbf16>

  // RHS test data
  %rhs_cst = arith.constant dense<[[ 0.6,  1.3,  0.1, -0.9],
                                   [ 0.5,  1.6,  1.8,  1.6],
                                   [-0.2,  0.4,  1.0,  0.4],
                                   [-1.3, -0.2, -2.2,  0.3]]> : vector<4x4xbf16>

  %rhs_mem = memref.alloca() : memref<4x4xbf16>
  vector.transfer_write %rhs_cst, %rhs_mem[%c0, %c0] {in_bounds = [true, true] } : vector<4x4xbf16>, memref<4x4xbf16>
  %rhs = vector.transfer_read %rhs_mem[%c0, %c0], %c0_bf16 {in_bounds = [true, true]} : memref<4x4xbf16>, vector<4x4xbf16>

  // Vector by matrix multiplication and accumulate with transposed RHS.
  %0 = vector.contract { indexing_maps = [
                           affine_map<(n, k) -> (k)>,
                           affine_map<(n, k) -> (n, k)>,
                           affine_map<(n, k) -> (n)>
                         ],
                         iterator_types = ["parallel", "reduction"],
                         kind = #vector.kind<add>
                       }
    %lhs, %rhs, %acc : vector<4xbf16>, vector<4x4xbf16> into vector<4xf32>

  // Display the result of the multiplication
  vector.print str "Result(BFMMLA, vecmat):\n"
  vector.print %0 : vector<4xf32>
  
  return
}

func.func @main() {
  // CHECK-LABEL: Result(BFMMLA):
  // CHECK: (  0.411922, 2.63254,  -0.219259,  3.89965 )
  // CHECK: ( -0.316515, 0.196875,  0.879375,  1.80924 )
  // CHECK: (  1.56867,  0.101367, -1.2784,   -1.41579 )
  // CHECK: ( -1.56041, -4.30078,   0.0196488, 1.88269 )
  func.call @matrix_by_matrix_mul_and_acc() : () -> ()

  // CHECK-LABEL: Result(BFMMLA, vecmat):
  // CHECK: ( 0.411922, 2.63254, -0.219259, 3.89965 )
  func.call @vector_by_matrix_mul_and_acc() : () -> ()

  return
}
