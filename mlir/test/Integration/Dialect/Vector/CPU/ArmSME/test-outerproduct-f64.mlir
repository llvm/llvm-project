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

llvm.func @printCString(!llvm.ptr<i8>)

func.func @printTileBegin() attributes { enable_arm_streaming_ignore } {
  %0 = llvm.mlir.addressof @str_tile_begin : !llvm.ptr<array<11 x i8>>
  %1 = llvm.mlir.constant(0 : index) : i64
  %2 = llvm.getelementptr %0[%1, %1]
    : (!llvm.ptr<array<11 x i8>>, i64, i64) -> !llvm.ptr<i8>
  llvm.call @printCString(%2) : (!llvm.ptr<i8>) -> ()
  return
}

func.func @printTileEnd() attributes { enable_arm_streaming_ignore } {
  %0 = llvm.mlir.addressof @str_tile_end : !llvm.ptr<array<9 x i8>>
  %1 = llvm.mlir.constant(0 : index) : i64
  %2 = llvm.getelementptr %0[%1, %1]
    : (!llvm.ptr<array<9 x i8>>, i64, i64) -> !llvm.ptr<i8>
  llvm.call @printCString(%2) : (!llvm.ptr<i8>) -> ()
  return
}

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
  func.call @printTileBegin() : () -> ()
  vector.print %tile : vector<[2]x[2]xf64>
  func.call @printTileEnd() : () -> ()

  return
}

llvm.mlir.global internal constant @str_tile_begin("TILE BEGIN\0A")
llvm.mlir.global internal constant @str_tile_end("TILE END\0A")
