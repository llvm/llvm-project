// DEFINE: %{compile} = mlir-opt %s  -transform-interpreter -test-transform-dialect-erase-schedule\
// DEFINE:    -cse -canonicalize -convert-vector-to-scf -arm-sve-legalize-vector-storage\
// DEFINE:    -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm -o %t
// DEFINE: %{entry} =
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e=%{entry} -entry-point-result=void --march=aarch64 --mattr="+sve" -shared-libs=%native_mlir_c_runner_utils

// This check whether the files compiles and generates a temporary that will be executed further down.
// RUN: %{compile}

// REDEFINE: %{entry} = matmul_i32
// RUN: %{run} | FileCheck %s --check-prefix=I32

// REDEFINE: %{entry} = matmul_f32
// RUN: %{run} | FileCheck %s --check-prefix=F32

// REDEFINE: %{entry} = dot_product_i32
// RUN: %{run} | FileCheck %s --check-prefix=DP

// REDEFINE: %{entry} = matvec_i32
// RUN: %{run} | FileCheck %s --check-prefix=MV

// NOTE: These tests are meant to complement the integration tests from:
//    * ../test-contraction.mlir
// (tests with fixed width vectors). Rather than duplicating those tests, this
// file focuses on excercissing scalable vectors in a few most common cases.

// TODO: Masks

#dotp_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#dotp_trait = {
  indexing_maps = #dotp_accesses,
  iterator_types = ["reduction"]
}

#matvec_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#matvec_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// Contraction: dot-product a x b.
func.func @dot_product_i32() {
  %acc = arith.constant 0: i32

  %vector_a = arith.constant dense<123> : vector<[4]xi32>
  %vector_b = arith.constant dense<314> : vector<[4]xi32>
  %vector_c = arith.constant dense<0> : vector<[4]xi32>

  // DOT PRODUCT 1
  %dp1 = vector.contract #dotp_trait %vector_a, %vector_b, %acc
    : vector<[4]xi32>, vector<[4]xi32> into i32
  // Dot product should be:
  //   * val = (123 * 314) * 4 * vscale,
  // so ...
  %vscale = vector.vscale
  %vscale_i32 = arith.index_cast %vscale : index to i32
  %dp1_div = arith.divui %dp1, %vscale_i32 : i32
  // ... val / vscale = 123 * 314 * 4 = 154488
  // DP: 154488
  vector.print %dp1_div : i32

  // DOT PRODUCT 2
  // The result of this dot-product should be 0.
  %dp2 = vector.contract #dotp_trait %vector_a, %vector_c, %acc
    : vector<[4]xi32>, vector<[4]xi32> into i32
  // DP: 0
  vector.print %dp2 : i32

  // DP: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT"

  return
}

// Contraction: matrix-vector A x c
func.func @matvec_i32() {
  %acc = arith.constant dense<0>: vector<3xi32>

  %vector_a = arith.constant dense<123> : vector<3x[4]xi32>
  %vector_b = arith.constant dense<314> : vector<[4]xi32>
  %vector_c = arith.constant dense<0> : vector<[4]xi32>

  // MATVEC 1
  %mv1 = vector.contract #matvec_trait %vector_a, %vector_b, %acc
    : vector<3x[4]xi32>, vector<[4]xi32> into vector<3xi32>
  // Every element in the output vector is a result of a dot product, for
  // which:
  //    val = (123 * 314) * 4 * vscale
  // so ...
  %vscale = vector.vscale
  %vscale_v = vector.splat %vscale : vector<3xindex>
  %vscale_i32 = arith.index_cast %vscale_v : vector<3xindex> to vector<3xi32>
  %mv1_div = arith.divui %mv1, %vscale_i32 : vector<3xi32>
  // ... val / vscale = 123 * 314 * 4 = 154488
  // MV: 154488, 154488, 154488
  vector.print %mv1_div : vector<3xi32>

  // MATVEC 2
  // The result of this matvec should be a vector of 0s.
  %mv2 = vector.contract #matvec_trait %vector_a, %vector_c, %acc
    : vector<3x[4]xi32>, vector<[4]xi32> into vector<3xi32>
  // MV: 0, 0, 0
  vector.print %mv2 : vector<3xi32>

  // MV: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT"

  return
}

func.func @matmul_i32() {
  // Setup vector A:
  %vector_a = arith.constant dense<123> : vector<3x5xi32>

  // Setup vector B:
  %vector_b = arith.constant dense<123> : vector<5x[2]xi32>

  // Setup vector C:
  %vector_c = arith.constant dense<314> : vector<3x[2]xi32>

  // Matmul
  %0 = vector.contract #matmat_trait %vector_a, %vector_b, %vector_c
    : vector<3x5xi32>, vector<5x[2]xi32> into vector<3x[2]xi32>

  // Print the output
  %slice1 = vector.extract %0[0] : vector<[2]xi32> from vector<3x[2]xi32>
  // I32: ( 75959, 75959
  vector.print %slice1 : vector<[2]xi32>
  %slice2 = vector.extract %0[1] : vector<[2]xi32> from vector<3x[2]xi32>
  // I32-NEXT: ( 75959, 75959
  vector.print %slice2 : vector<[2]xi32>
  %slice3 = vector.extract %0[2] : vector<[2]xi32> from vector<3x[2]xi32>
  // I32-NEXT: ( 75959, 75959
  vector.print %slice3 : vector<[2]xi32>

  // CHECK: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT"

  return
}

func.func @matmul_f32() {
  // Setup vector A:
  %vector_a = arith.constant dense<1.23> : vector<3x5xf32>

  // Setup vector B:
  %vector_b = arith.constant dense<1.23> : vector<5x[2]xf32>

  // Setup vector C:
  %vector_c = arith.constant dense<3.14> : vector<3x[2]xf32>

  // Matmul
  %0 = vector.contract #matmat_trait %vector_a, %vector_b, %vector_c
    : vector<3x5xf32>, vector<5x[2]xf32> into vector<3x[2]xf32>

  // Print the output
  %slice1 = vector.extract %0[0] : vector<[2]xf32> from vector<3x[2]xf32>
  // F32: ( 10.7045, 10.7045
  vector.print %slice1 : vector<[2]xf32>
  %slice2 = vector.extract %0[1] : vector<[2]xf32> from vector<3x[2]xf32>
  // F32-NEXT: ( 10.7045, 10.7045
  vector.print %slice2 : vector<[2]xf32>
  %slice3 = vector.extract %0[2] : vector<[2]xf32> from vector<3x[2]xf32>
  // F32-NEXT: ( 10.7045, 10.7045
  vector.print %slice3 : vector<[2]xf32>

  // CHECK: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT"

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.yield
  }
}
