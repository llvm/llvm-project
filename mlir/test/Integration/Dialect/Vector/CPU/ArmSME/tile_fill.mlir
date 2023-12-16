// RUN: mlir-opt %s -enable-arm-streaming="streaming-mode=streaming-locally za-mode=new-za" \
// RUN:   -convert-vector-to-arm-sme -convert-arm-sme-to-scf -allocate-arm-sme-tiles \
// RUN:   -convert-arm-sme-to-llvm -cse -canonicalize \
// RUN:   -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd \
// RUN:  -march=aarch64 -mattr=+sve,+sme \
// RUN:  -e entry -entry-point-result=i32 \
// RUN:  -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%arm_sme_abi_shlib | \
// RUN: FileCheck %s

// Integration test demonstrating filling a 32-bit element ZA tile with a
// non-zero constant via vector to tile (MOVA) ops.

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
  vector.print str "TILE BEGIN"
  vector.print %tile : vector<[4]x[4]xi32>
  vector.print str "TILE END"

  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
