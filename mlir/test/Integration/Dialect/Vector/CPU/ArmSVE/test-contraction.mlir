// DEFINE: %{compile} = mlir-opt %s  -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule\
// DEFINE:    -cse -canonicalize -convert-vector-to-scf -arm-sve-legalize-vector-storage\
// DEFINE:    -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm -o %t
// DEFINE: %{entry} =
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e=%{entry} -entry-point-result=void --march=aarch64 --mattr="+sve" -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext

// This check whether the files compiles and generates a temporary that will be executed further down.
// RUN: %{compile}

// REDEFINE: %{entry} = matmul_i32
// RUN: %{run} | FileCheck %s --check-prefix=I32

// REDEFINE: %{entry} = matmul_f32
// RUN: %{run} | FileCheck %s --check-prefix=F32

// NOTE: These tests are meant to complement the integration tests from:
//    * ../test-contraction.mlir
// (tests with fixed width vectors). Rather than duplicating those tests, this
// file focuses on excercissing scalable vectors in a few most common cases.

// TODO: Masks + matvec + dot product

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
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

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %f = transform.structured.match ops{["func.func"]} in %module_op
    : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
  } : !transform.any_op
}
