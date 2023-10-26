// DEFINE: %{entry_point} = test_outerproduct_no_accumulator_4x4xf32
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -enable-arm-streaming="mode=locally enable-za" \
// DEFINE:   -convert-vector-to-arm-sme -convert-arm-sme-to-scf \
// DEFINE:   -convert-vector-to-llvm="enable-arm-sme" -cse -canonicalize \
// DEFINE:   -allocate-arm-sme-tiles -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:   -march=aarch64 -mattr=+sve,+sme \
// DEFINE:   -e %{entry_point} -entry-point-result=void \
// DEFINE:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s --check-prefix=WITHOUT-ACC

// REDEFINE: %{entry_point} = test_outerproduct_with_accumulator_4x4xf32
// RUN: %{compile} | %{run} | FileCheck %s --check-prefix=WITH-ACC

func.func @test_outerproduct_no_accumulator_4x4xf32() {
  %c0 = arith.constant 0 : index

  %vector_i32 = llvm.intr.experimental.stepvector : vector<[4]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[4]xi32> to vector<[4]xf32>
  %tile = vector.outerproduct %vector, %vector : vector<[4]xf32>, vector<[4]xf32>

  // Print the tile. The smallest SVL is 128-bits so the tile will be at least
  // 4x4xf32.
  //
  // WITHOUT-ACC:      TILE BEGIN
  // WITHOUT-ACC-NEXT: ( 0, 0, 0, 0
  // WITHOUT-ACC-NEXT: ( 0, 1, 2, 3
  // WITHOUT-ACC-NEXT: ( 0, 2, 4, 6
  // WITHOUT-ACC-NEXT: ( 0, 3, 6, 9
  // WITHOUT-ACC:      TILE END
  vector.print str "TILE BEGIN"
  vector.print %tile : vector<[4]x[4]xf32>
  vector.print str "TILE END"

  return
}

func.func @test_outerproduct_with_accumulator_4x4xf32() {
  %c0 = arith.constant 0 : index
  %f10 = arith.constant 10.0 : f32

  %acc = vector.broadcast %f10 : f32 to vector<[4]x[4]xf32>
  %vector_i32 = llvm.intr.experimental.stepvector : vector<[4]xi32>
  %vector = arith.sitofp %vector_i32 : vector<[4]xi32> to vector<[4]xf32>
  %tile = vector.outerproduct %vector, %vector, %acc : vector<[4]xf32>, vector<[4]xf32>

  // Print the tile. The smallest SVL is 128-bits so the tile will be at least
  // 4x4xf32.
  //
  // WITH-ACC:      TILE BEGIN
  // WITH-ACC-NEXT: ( 10, 10, 10, 10
  // WITH-ACC-NEXT: ( 10, 11, 12, 13
  // WITH-ACC-NEXT: ( 10, 12, 14, 16
  // WITH-ACC-NEXT: ( 10, 13, 16, 19
  // WITH-ACC:      TILE END
  vector.print str "TILE BEGIN"
  vector.print %tile : vector<[4]x[4]xf32>
  vector.print str "TILE END"

  return
}
