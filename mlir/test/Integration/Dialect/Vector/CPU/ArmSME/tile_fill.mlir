// RUN: mlir-opt %s -enable-arm-streaming="mode=locally enable-za" \
// RUN:   -convert-vector-to-arm-sme -convert-arm-sme-to-scf \
// RUN:   -convert-vector-to-llvm="enable-arm-sme" -cse -canonicalize \
// RUN:   -allocate-arm-sme-tiles -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd \
// RUN:  -march=aarch64 -mattr=+sve,+sme \
// RUN:  -e entry -entry-point-result=i32 \
// RUN:  -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | \
// RUN: FileCheck %s

// Integration test demonstrating filling a 32-bit element ZA tile with a
// non-zero constant via vector to tile (MOVA) ops.

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

func.func @entry() -> i32 {
  // Fill a tile with '123'. This will get lowered to a 1-d vector splat of
  // '123' and a loop that writes this vector to each tile slice in the ZA
  // tile.
  %tile = arith.constant dense<123> : vector<[4]x[4]xi32>

  // Print the tile. The smallest SVL is 128-bits so the tile will be at least
  // 4x4xi32.
  //
  // CHECK:      TILE BEGIN
  // CHECK-NEXT: ( 123, 123, 123, 123
  // CHECK-NEXT: ( 123, 123, 123, 123
  // CHECK-NEXT: ( 123, 123, 123, 123
  // CHECK-NEXT: ( 123, 123, 123, 123
  // CHECK:      TILE END
  func.call @printTileBegin() : () -> ()
  vector.print %tile : vector<[4]x[4]xi32>
  func.call @printTileEnd() : () -> ()

  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}

llvm.mlir.global internal constant @str_tile_begin("TILE BEGIN\0A")
llvm.mlir.global internal constant @str_tile_end("TILE END\0A")
