// DEFINE: %{entry_point} = entry
// DEFINE: %{compile} = mlir-opt %s -test-lower-to-arm-sme -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:   -march=aarch64 -mattr=+sve,+sme \
// DEFINE:   -e %{entry_point} -entry-point-result=void \
// DEFINE:   -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils,%native_arm_sme_abi_shlib

// RUN: %{compile} | %{run} | FileCheck %s

func.func @entry() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32

  // Calculate the size of a 32-bit tile, e.g. ZA{n}.s.
  %svl_s = arm_sme.streaming_vl <word>
  %za_s_size = arith.muli %svl_s, %svl_s : index

  // Allocate memory.
  %mem1 = memref.alloca(%za_s_size) : memref<?xi32>

  // Fill each "row" of "mem1" with row number.
  //
  // For example, assuming an SVL of 128-bits:
  //
  //   0, 0, 0, 0
  //   1, 1, 1, 1
  //   2, 2, 2, 2
  //   3, 3, 3, 3
  //
  %init_0 = arith.constant 0 : i32
  scf.for %i = %c0 to %za_s_size step %svl_s iter_args(%val = %init_0) -> (i32) {
    %splat_val = vector.broadcast %val : i32 to vector<[4]xi32>
    vector.store %splat_val, %mem1[%i] : memref<?xi32>, vector<[4]xi32>
    %val_next = arith.addi %val, %c1_i32 : i32
    scf.yield %val_next : i32
  }

  // Load tile from "mem1" vertically.
  %0 = arm_sme.tile_load %mem1[%c0, %c0] layout<vertical> : memref<?xi32>, vector<[4]x[4]xi32>

  // 1. ORIGINAL HORIZONTAL LAYOUT
  // Dump "mem1". The smallest SVL is 128-bits so the tile will be at least
  // 4x4xi32.
  //
  // CHECK:      TILE BEGIN
  // CHECK-NEXT: ( 0, 0, 0, 0
  // CHECK-NEXT: ( 1, 1, 1, 1
  // CHECK-NEXT: ( 2, 2, 2, 2
  // CHECK-NEXT: ( 3, 3, 3, 3
  // CHECK:      TILE END
  vector.print str "TILE BEGIN\n"
  scf.for %i = %c0 to %za_s_size step %svl_s {
    %tileslice = vector.load %mem1[%i] : memref<?xi32>, vector<[4]xi32>
    vector.print %tileslice : vector<[4]xi32>
  }
  vector.print str "TILE END\n"

  // 2. VERTICAL LAYOUT
  // Dump "mem2". The smallest SVL is 128-bits so the tile will be at least
  // 4x4xi32.
  //
  // CHECK:      TILE BEGIN
  // CHECK-NEXT: ( 0, 1, 2, 3
  // CHECK-NEXT: ( 0, 1, 2, 3
  // CHECK-NEXT: ( 0, 1, 2, 3
  // CHECK-NEXT: ( 0, 1, 2, 3
  // CHECK:      TILE END
  vector.print str "TILE BEGIN\n"
  vector.print %0 : vector<[4]x[4]xi32>
  vector.print str "TILE END\n"

  return
}
