// REQUIRES: arm-emulator

// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --convert-vector-to-scf --convert-scf-to-cf  --convert-vector-to-llvm='enable-arm-sve enable-arm-bf16' \
// DEFINE:   --expand-strided-metadata --convert-to-llvm --finalize-memref-to-llvm  \
// DEFINE:   --lower-affine --convert-arith-to-llvm --reconcile-unrealized-casts \
// DEFINE: -o %t

// DEFINE: %{entry_point} = main

// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void  --march=aarch64 --mattr="+sve,+bf16" \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_arm_runner_utils

// RUN: rm -f %t && %{compile} && FileCheck %s --input-file=%t -check-prefix CHECK-IR && %{run} | FileCheck %s

#packed_maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

//
// Test the lowering of `vector.contract` using the `LowerContractionToSVEBFMMLAPattern`
//
// The operation that the `vector.contract` in this test performs is matrix
// multiplication with accumulate
//     OUT = ACC + LHS * RHS
// of two BFloat16 matrices LHS and RHS, and a Float32 matrix ACC into a Float32 OUT.
//
// Tested are calculations as well as that the relevant `ArmSVE` dialect
// operation ('arm_sve.intr.bfmmla`) is emitted.
//
// That pattern above handles (therefore this test prepares) input/output vectors with
// specific shapes:
//   * LHS:      vector<Mx4xbf16>
//   * RHS:      vector<[N]x4xbf16>
//   * ACC, OUT: vector<Mx[N]xf32>
// Note that the RHS is transposed.
// This data layout makes it efficient to load data into SVE
// registers in the layout expected by te BFMMLA instruction.
// Such a `vector.contract` is representative of the code we aim to generate
// by scalable vectorisation of `linalg.mmt4d`.
// See mlir/lib/Dialect/ArmSVE/Transforms/LowerContractToSVEPatterns.cpp
// for more information and rationale about these shapes.
//
// In this specific test we use M == 4 and N == 4
//

// Allocate and initialise a memref containing test data for use as the ACC
// operand. The memref has one dynamic dimension whose extent depends on the
// runtime value of VSCALE.
//
// The input parameter `%in` is a vector that is replicated VSCALE times
// across the columns of the memref.
func.func private @prepareAccTestData(%in: vector<4x4xf32>) -> memref<4x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  %vs = vector.vscale
  %nCols = arith.muli %c4, %vs : index
  %mem = memref.alloc(%nCols) : memref<4x?xf32>

  scf.for %j = %c0 to %nCols step %c4 {
    vector.transfer_write %in, %mem[%c0, %j] {in_bounds = [true, true]} : vector<4x4xf32>, memref<4x?xf32>
  }

  return %mem : memref<4x?xf32>
}

// Allocate and initialise a memref containing test data for use as the LHS
// operand. This function just writes the parameter `%in` into the memref.
// The size of the LHS does not depends on VSCALE.
func.func private @prepareLHSTestData(%in: vector<4x4xbf16>) -> memref<4x4xbf16> {
  %c0 = arith.constant 0 : index

  %mem = memref.alloc() : memref<4x4xbf16>
  vector.transfer_write %in, %mem[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xbf16>, memref<4x4xbf16>

  return %mem : memref<4x4xbf16>
}

// Allocate and initialise a memref containing test data for use as the RHS
// operand. The memref has one dynamic dimension whose extent depends on the
// runtime value of VSCALE.
//
// The input parameter `%in` is a vector that is replicated VSCALE times
// across the rows of the memref.
//
// For convenience, flatten the memref, since the RHS vector is read first as a
// single-dimensional scalable vector and then cast into [N]x4 shape.
func.func private @prepareRHSTestData(%in: vector<4x4xbf16>) -> memref<?xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  %vs = vector.vscale
  %nRows = arith.muli %c4, %vs : index
  %mem = memref.alloc(%nRows) : memref<?x4xbf16>

  scf.for %i = %c0 to %nRows step %c4 {
    vector.transfer_write %in, %mem[%i, %c0] {in_bounds = [true, true]} : vector<4x4xbf16>, memref<?x4xbf16>
  }

  %mem_out = memref.collapse_shape %mem [[0, 1]] : memref<?x4xbf16> into memref<?xbf16>
  return %mem_out : memref<?xbf16>
}


// CHECK-IR-LABEL: llvm.func @test_bfmmla
// CHECK-IR-COUNT-4: arm_sve.intr.bfmmla
func.func @test_bfmmla() {

  %c0 = arith.constant 0 : index
  %c0_f32 = arith.constant 0.0 : f32
  %c0_bf16 = arith.constant 0.0 : bf16

  // Accumulator test data
  %acc_cst = arith.constant dense<[[ 0.7,  1.0, -0.1,  1.8],
                                   [-0.5,  0.9,  0.7, -0.7],
                                   [ 0.5, -1.3, -2.2,  0.1],
                                   [-0.7,  1.0,  1.7, -1.0]]> : vector<4x4xf32>

  %acc_mem = func.call @prepareAccTestData(%acc_cst) : (vector<4x4xf32>) -> memref<4x?xf32>
  %acc = vector.transfer_read %acc_mem[%c0, %c0], %c0_f32 {in_bounds = [true, true]} : memref<4x?xf32>, vector<4x[4]xf32>

  // LHS test data
  %lhs_cst = arith.constant dense<[[ 0.1,  0.7, -0.9,  1.3],
                                   [-1.6,  0.7, -0.3, -0.3],
                                   [-0.4,  0.6,  0.8, -0.5],
                                   [-0.6, -1.0, -1.0, -1.0]]> : vector<4x4xbf16>

  %lhs_mem = func.call @prepareLHSTestData(%lhs_cst) : (vector<4x4xbf16>) -> memref<4x4xbf16>
  %lhs = vector.transfer_read %lhs_mem[%c0, %c0], %c0_bf16 {in_bounds = [true, true]} : memref<4x4xbf16>, vector<4x4xbf16>

  // RHS test data
  %rhs_cst = arith.constant dense<[[ 0.6,  1.3,  0.1, -0.9],
                                   [ 0.5,  1.6,  1.8,  1.6],
                                   [-0.2,  0.4,  1.0,  0.4],
                                   [-1.3, -0.2, -2.2,  0.3]]> : vector<4x4xbf16>

  %rhs_mem = func.call @prepareRHSTestData(%rhs_cst) : (vector<4x4xbf16>) -> memref<?xbf16>
  %rhs_flat = vector.transfer_read %rhs_mem[%c0], %c0_bf16 {in_bounds = [true]} :  memref<?xbf16>, vector<[16]xbf16>
  %rhs = vector.shape_cast %rhs_flat : vector<[16]xbf16> to vector<[4]x4xbf16>

  // Matrix multiplication and accumulate with transposed RHS.
  %0 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<4x4xbf16>, vector<[4]x4xbf16> into vector<4x[4]xf32>

  // Display the result of the multiplication
  vector.print str "Result(BFMMLA):\n"
  %u0 = vector.extract %0[0] : vector<[4]xf32> from vector<4x[4]xf32>
  %u1 = vector.extract %0[1] : vector<[4]xf32> from vector<4x[4]xf32>
  %u2 = vector.extract %0[2] : vector<[4]xf32> from vector<4x[4]xf32>
  %u3 = vector.extract %0[3] : vector<[4]xf32> from vector<4x[4]xf32>
  vector.print %u0 : vector<[4]xf32>
  vector.print %u1 : vector<[4]xf32>
  vector.print %u2 : vector<[4]xf32>
  vector.print %u3 : vector<[4]xf32>

  // Deallocate the buffers.
  memref.dealloc %acc_mem : memref<4x?xf32>
  memref.dealloc %lhs_mem : memref<4x4xbf16>
  memref.dealloc %rhs_mem : memref<?xbf16>

  return
}

// Perform each test with SVE vector lengths 128 bits and 256 bits (i.e. VSCALEs
// 1 and 2, respectively). The vector length is set via the `setArmVLBits`
// function. The effect of setting a different vector length is that the tests
// allocate and operate on different sized buffers (see `prepare<X>TestData`
// functions).

func.func @main() {
  %c128 = arith.constant 128 : i32
  %c256 = arith.constant 256 : i32

// CHECK-LABEL: Result(BFMMLA):
// CHECK: (  0.411922, 2.63254,  -0.219259,  3.89965 )
// CHECK: ( -0.316515, 0.196875,  0.879375,  1.80924 )
// CHECK: (  1.56867,  0.101367, -1.2784,   -1.41579 )
// CHECK: ( -1.56041, -4.30078,   0.0196488, 1.88269 )
  func.call @setArmVLBits(%c128) : (i32) -> ()
  func.call @test_bfmmla() : () -> ()

// CHECK: Result(BFMMLA):
// CHECK: (  0.411922, 2.63254,  -0.219259,  3.89965,  0.411922, 2.63254,  -0.219259,  3.89965 )
// CHECK: ( -0.316515, 0.196875,  0.879375,  1.80924, -0.316515, 0.196875,  0.879375,  1.80924 )
// CHECK: (  1.56867,  0.101367, -1.2784,   -1.41579,  1.56867,  0.101367, -1.2784,   -1.41579 )
// CHECK: ( -1.56041, -4.30078,   0.0196488, 1.88269, -1.56041, -4.30078,   0.0196488, 1.88269 )
  func.call @setArmVLBits(%c256) : (i32) -> ()
  func.call @test_bfmmla() : () -> ()

  return
}

func.func private @setArmVLBits(%bits : i32)
func.func private @printMemrefF32(%ptr : memref<*xf32>)
