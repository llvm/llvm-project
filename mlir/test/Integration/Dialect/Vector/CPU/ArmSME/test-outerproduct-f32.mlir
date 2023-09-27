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
  func.call @printTileBegin() : () -> ()
  vector.print %tile : vector<[4]x[4]xf32>
  func.call @printTileEnd() : () -> ()

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
  func.call @printTileBegin() : () -> ()
  vector.print %tile : vector<[4]x[4]xf32>
  func.call @printTileEnd() : () -> ()

  return
}

llvm.mlir.global internal constant @str_tile_begin("TILE BEGIN\0A")
llvm.mlir.global internal constant @str_tile_end("TILE END\0A")
