// DEFINE: %{entry_point} = test_outerproduct_with_accumulator_2x2xf64
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   -enable-arm-streaming="mode=locally enable-za" \
// DEFINE:   -convert-vector-to-arm-sme -convert-arm-sme-to-scf \
// DEFINE:   -convert-vector-to-llvm="enable-arm-sme" -cse -canonicalize \
// DEFINE:   -allocate-arm-sme-tiles -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:   -march=aarch64 -mattr=+sve,+sme-f64f64 \
// DEFINE:   -e %{entry_point} -entry-point-result=void \
// DEFINE:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s

func.func @test_outerproduct_with_accumulator_2x2xf64() {
  %f1 = arith.constant 1.0 : f64
  %f2 = arith.constant 2.0 : f64
  %f10 = arith.constant 10.0 : f64

  %a = vector.splat %f1 : vector<[2]xf64>
  %b = vector.splat %f2 : vector<[2]xf64>
  // TODO: vector.splat doesn't support ArmSME.
  %c = vector.broadcast %f10 : f64 to vector<[2]x[2]xf64>

  %tile = vector.outerproduct %a, %b, %c : vector<[2]xf64>, vector<[2]xf64>

  // Print the tile. The smallest SVL is 128-bits so the tile will be at least
  // 2x2xf64.
  //
  // CHECK:      TILE BEGIN
  // CHECK-NEXT: ( 12, 12
  // CHECK-NEXT: ( 12, 12
  // CHECK:      TILE END
  vector.print str "TILE BEGIN"
  vector.print %tile : vector<[2]x[2]xf64>
  vector.print str "TILE END"

  return
}
