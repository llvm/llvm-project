// RUN: mlir-opt %s -allocate-arm-sme-tiles -split-input-file -verify-diagnostics | \
// RUN: FileCheck %s  --check-prefix=AFTER-TILE-ALLOC
// RUN: mlir-opt %s -allocate-arm-sme-tiles -convert-arm-sme-to-llvm -canonicalize -cse \
// RUN:   -split-input-file -verify-diagnostics | \
// RUN: FileCheck %s  --check-prefix=AFTER-LLVM-LOWERING

// -----

/// Checks tile spill/reloads are inserted around in-memory tiles (i.e. tiles
/// that were not assigned a physical SME tile).
///
/// These spills are currently very naive and paranoid and will spill/reload
/// entire tiles around ArmSME ops.
///
/// The general pattern is:
///
/// During tile allocation if there's not a physical tile ID available an op
/// will be assigned an in-memory tile ID (which is a tile ID >= 16).
///
/// Example:
///
///   arm_sme.zero : vector<[8]x[8]xi16>
///
/// Becomes:
///
///   arm_sme.zero { tile_id = 16 } : vector<[8]x[8]xi16>
///
/// This works like normal till the final lowering to LLVM, where spills and
/// reloads will be inserted around uses of in-memory tiles.
///
/// So the above example becomes:
///
/// // Placed at the top of the function:
/// %tileAlloca = memref.alloca(%svl_h, %svl_h) : memref<?x?xi16>
///
/// Then around the op:
///
/// // Swap contents of %tileAlloca and tile 0
/// scf.for %sliceIdx ... {
///   %currentSlice = arm_sme.intr.read.horiz {tile_id = 0}
///   arm_sme.intr.ld1h.horiz %tileAlloca[%sliceIdx, %c0] {tile_id = 0}
///   vector.store %currentSlice, %tileAlloca[%sliceIdx, %c0]
/// }
/// // Execute the op using tile 0
/// arm_sme.intr.zero
/// // Swap contents of %tileAlloca and tile 0
/// scf.for %sliceIdx ... {
///   %currentSlice = arm_sme.intr.read.horiz {tile_id = 0}
///   arm_sme.intr.ld1h.horiz %tileAlloca[%sliceIdx, %c0] {tile_id = 0}
///   vector.store %currentSlice, %tileAlloca[%sliceIdx, %c0]
/// }
///

func.func @use_too_many_tiles() {
  %0 = arm_sme.zero : vector<[4]x[4]xi32>
  %1 = arm_sme.zero : vector<[4]x[4]xi32>
  // expected-warning @below {{failed to allocate physical tile to operation, all tile operations will go through memory, expect performance degradation}}
  %2 = arm_sme.zero : vector<[8]x[8]xi16>
  return
}
// AFTER-TILE-ALLOC-LABEL: @use_too_many_tiles
//      AFTER-TILE-ALLOC: arm_sme.zero
// AFTER-TILE-ALLOC-SAME:   tile_id = 0
//      AFTER-TILE-ALLOC: arm_sme.zero
// AFTER-TILE-ALLOC-SAME:   tile_id = 1
//      AFTER-TILE-ALLOC: arm_sme.zero
// AFTER-TILE-ALLOC-SAME:   tile_id = 16

// AFTER-LLVM-LOWERING-LABEL: @use_too_many_tiles
//  AFTER-LLVM-LOWERING-DAG: %[[C0:.*]] = arith.constant 0 : index
//  AFTER-LLVM-LOWERING-DAG: %[[C1:.*]] = arith.constant 1 : index
//  AFTER-LLVM-LOWERING-DAG: %[[C8:.*]] = arith.constant 8 : index
//  AFTER-LLVM-LOWERING-DAG: %[[VSCALE:.*]] = vector.vscale
//  AFTER-LLVM-LOWERING-DAG: %[[SVL_H:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index
//  AFTER-LLVM-LOWERING-DAG: %[[TILE_ALLOCA:.*]] = memref.alloca(%[[SVL_H]], %[[SVL_H]])
// AFTER-LLVM-LOWERING-SAME:   {arm_sme.in_memory_tile_id = 16 : i32} : memref<?x?xi16>
//
//  AFTER-LLVM-LOWERING-NOT: scf.for
//      AFTER-LLVM-LOWERING: arm_sme.intr.zero
//
//  AFTER-LLVM-LOWERING-NOT: scf.for
//      AFTER-LLVM-LOWERING: arm_sme.intr.zero
//
//      AFTER-LLVM-LOWERING: scf.for
// AFTER-LLVM-LOWERING-SAME: %[[C0]] to %[[SVL_H]] step %[[C1]] {
//      AFTER-LLVM-LOWERING:   arm_sme.intr.read.horiz
// AFTER-LLVM-LOWERING-NEXT:   arm_sme.intr.ld1h.horiz
// AFTER-LLVM-LOWERING-NEXT:   vector.store
// AFTER-LLVM-LOWERING-NEXT: }
//      AFTER-LLVM-LOWERING: arm_sme.intr.zero
//      AFTER-LLVM-LOWERING: scf.for
// AFTER-LLVM-LOWERING-SAME: %[[C0]] to %[[SVL_H]] step %[[C1]] {
//      AFTER-LLVM-LOWERING:   arm_sme.intr.read.horiz
// AFTER-LLVM-LOWERING-NEXT:   arm_sme.intr.ld1h.horiz
// AFTER-LLVM-LOWERING-NEXT:   vector.store
// AFTER-LLVM-LOWERING-NEXT: }
