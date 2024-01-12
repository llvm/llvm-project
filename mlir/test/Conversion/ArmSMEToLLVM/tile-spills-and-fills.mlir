// RUN: mlir-opt %s -allocate-arm-sme-tiles -split-input-file -verify-diagnostics | \
// RUN: FileCheck %s  --check-prefix=AFTER-TILE-ALLOC
// RUN: mlir-opt %s -allocate-arm-sme-tiles -convert-arm-sme-to-llvm -canonicalize -cse \
// RUN:   -split-input-file -verify-diagnostics | \
// RUN: FileCheck %s  --check-prefix=AFTER-LLVM-LOWERING

/// Checks tile spill/reloads are inserted around in-memory tiles (i.e. tiles
/// that were not assigned a physical SME tile).
///
/// These spills are currently very naive and will spill/reload entire tiles
/// around ArmSME ops.
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
/// This works like normal until the final lowering to LLVM, where spills and
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
/// scf.for %sliceIdx ...
///   %currentSlice = arm_sme.intr.read.horiz {tile_id = 0}
///   arm_sme.intr.ld1h.horiz %tileAlloca[%sliceIdx, %c0] {tile_id = 0}
///   vector.store %currentSlice, %tileAlloca[%sliceIdx, %c0]
/// // Execute the op using tile 0
/// arm_sme.intr.zero
/// // Swap contents of %tileAlloca and tile 0
/// scf.for %sliceIdx ...
///   %currentSlice = arm_sme.intr.read.horiz {tile_id = 0}
///   arm_sme.intr.ld1h.horiz %tileAlloca[%sliceIdx, %c0] {tile_id = 0}
///   vector.store %currentSlice, %tileAlloca[%sliceIdx, %c0]
///

// -----

/// Note: In this example loads into ZA are inserted before the zero instruction.
/// These are obviously redundant, but there's no checks to avoid this.
func.func @use_too_many_tiles() {
  %0 = arm_sme.zero : vector<[4]x[4]xi32>
  %1 = arm_sme.zero : vector<[4]x[4]xi32>
  // expected-warning @below {{failed to allocate SME virtual tile to operation, all tile operations will go through memory, expect degraded performance}}
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
//                           Note: 17 is the mask for the 32-bit tile 0.
//      AFTER-LLVM-LOWERING: "arm_sme.intr.zero"() <{tile_mask = 17 : i32}>
//
//  AFTER-LLVM-LOWERING-NOT: scf.for
//                           Note: 34 is the mask for the 32-bit tile 1.
//      AFTER-LLVM-LOWERING: "arm_sme.intr.zero"() <{tile_mask = 34 : i32}>
//
//      AFTER-LLVM-LOWERING: scf.for
// AFTER-LLVM-LOWERING-SAME: %[[C0]] to %[[SVL_H]] step %[[C1]] {
//      AFTER-LLVM-LOWERING:   %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[TILE_ALLOCA]]
//      AFTER-LLVM-LOWERING:   %[[BASE_PTR:.*]] = llvm.extractvalue %[[MEM_DESC]][1]
//      AFTER-LLVM-LOWERING:   %[[SLICE_PTR:.*]] = llvm.getelementptr %[[BASE_PTR]]
//      AFTER-LLVM-LOWERING:   %[[SLICE:.*]] = "arm_sme.intr.read.horiz"{{.*}} <{tile_id = 0 : i32}>
// AFTER-LLVM-LOWERING-NEXT:   "arm_sme.intr.ld1h.horiz"({{.*}}, %[[SLICE_PTR]], {{.*}}) <{tile_id = 0 : i32}>
// AFTER-LLVM-LOWERING-NEXT:   vector.store %[[SLICE]], %[[TILE_ALLOCA]]
// AFTER-LLVM-LOWERING-NEXT: }
//                           Note: 85 is the mask for the 16-bit tile 0.
//      AFTER-LLVM-LOWERING: "arm_sme.intr.zero"() <{tile_mask = 85 : i32}>
//      AFTER-LLVM-LOWERING: scf.for
// AFTER-LLVM-LOWERING-SAME: %[[C0]] to %[[SVL_H]] step %[[C1]] {
//      AFTER-LLVM-LOWERING:   %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[TILE_ALLOCA]]
//      AFTER-LLVM-LOWERING:   %[[BASE_PTR:.*]] = llvm.extractvalue %[[MEM_DESC]][1]
//      AFTER-LLVM-LOWERING:   %[[SLICE_PTR:.*]] = llvm.getelementptr %[[BASE_PTR]]
//      AFTER-LLVM-LOWERING:   %[[SLICE:.*]] = "arm_sme.intr.read.horiz"{{.*}} <{tile_id = 0 : i32}>
// AFTER-LLVM-LOWERING-NEXT:   "arm_sme.intr.ld1h.horiz"({{.*}}, %[[SLICE_PTR]], {{.*}}) <{tile_id = 0 : i32}>
// AFTER-LLVM-LOWERING-NEXT:   vector.store %[[SLICE]], %[[TILE_ALLOCA]]
// AFTER-LLVM-LOWERING-NEXT: }

// -----

/// Note: In this example an entire tile swap is inserted before/after the
/// `arm_sme.load_tile_slice` operation. Really, this only needs to spill a
/// single tile slice (and can omit the initial load, like in the previous example).
func.func @very_excessive_spills(%memref : memref<?x?xf32>) -> vector<[4]x[4]xf32> {
  %useAllTiles = arm_sme.get_tile : vector<[16]x[16]xi8>
  %c0 = arith.constant 0 : index
  // expected-warning @below {{failed to allocate SME virtual tile to operation, all tile operations will go through memory, expect degraded performance}}
  %tile = arm_sme.get_tile : vector<[4]x[4]xf32>
  %mask = vector.constant_mask [4] : vector<[4]xi1>
  %loadSlice = arm_sme.load_tile_slice %memref[%c0, %c0], %mask, %tile, %c0 : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
  return %loadSlice : vector<[4]x[4]xf32>
}
// AFTER-TILE-ALLOC-LABEL: @very_excessive_spills
//      AFTER-TILE-ALLOC: arm_sme.get_tile
// AFTER-TILE-ALLOC-SAME:   tile_id = 0
//      AFTER-TILE-ALLOC: arm_sme.load_tile_slice
// AFTER-TILE-ALLOC-SAME:   tile_id = 16

// AFTER-LLVM-LOWERING-LABEL: @very_excessive_spills
//  AFTER-LLVM-LOWERING-DAG: %[[C0:.*]] = arith.constant 0 : index
//  AFTER-LLVM-LOWERING-DAG: %[[C1:.*]] = arith.constant 1 : index
//  AFTER-LLVM-LOWERING-DAG: %[[C4:.*]] = arith.constant 4 : index
//  AFTER-LLVM-LOWERING-DAG: %[[VSCALE:.*]] = vector.vscale
//  AFTER-LLVM-LOWERING-DAG: %[[SVL_S:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
//  AFTER-LLVM-LOWERING-DAG: %[[TILE_ALLOCA:.*]] = memref.alloca(%[[SVL_S]], %[[SVL_S]])
// AFTER-LLVM-LOWERING-SAME:   {arm_sme.in_memory_tile_id = 16 : i32} : memref<?x?xf32>
//
//      AFTER-LLVM-LOWERING: scf.for
// AFTER-LLVM-LOWERING-SAME: %[[C0]] to %[[SVL_S]] step %[[C1]] {
//      AFTER-LLVM-LOWERING:   %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[TILE_ALLOCA]]
//      AFTER-LLVM-LOWERING:   %[[BASE_PTR:.*]] = llvm.extractvalue %[[MEM_DESC]][1]
//      AFTER-LLVM-LOWERING:   %[[SLICE_PTR:.*]] = llvm.getelementptr %[[BASE_PTR]]
//      AFTER-LLVM-LOWERING:   %[[SLICE:.*]] = "arm_sme.intr.read.horiz"{{.*}} <{tile_id = 0 : i32}>
// AFTER-LLVM-LOWERING-NEXT:   "arm_sme.intr.ld1w.horiz"({{.*}}, %[[SLICE_PTR]], {{.*}}) <{tile_id = 0 : i32}>
// AFTER-LLVM-LOWERING-NEXT:   vector.store %[[SLICE]], %[[TILE_ALLOCA]]
// AFTER-LLVM-LOWERING-NEXT: }
//      AFTER-LLVM-LOWERING: "arm_sme.intr.ld1w.horiz"{{.*}} <{tile_id = 0 : i32}>
//      AFTER-LLVM-LOWERING: scf.for
// AFTER-LLVM-LOWERING-SAME: %[[C0]] to %[[SVL_S]] step %[[C1]] {
//      AFTER-LLVM-LOWERING:   %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[TILE_ALLOCA]]
//      AFTER-LLVM-LOWERING:   %[[BASE_PTR:.*]] = llvm.extractvalue %[[MEM_DESC]][1]
//      AFTER-LLVM-LOWERING:   %[[SLICE_PTR:.*]] = llvm.getelementptr %[[BASE_PTR]]
//      AFTER-LLVM-LOWERING:   %[[SLICE:.*]] = "arm_sme.intr.read.horiz"{{.*}} <{tile_id = 0 : i32}>
// AFTER-LLVM-LOWERING-NEXT:   "arm_sme.intr.ld1w.horiz"({{.*}}, %[[SLICE_PTR]], {{.*}}) <{tile_id = 0 : i32}>
// AFTER-LLVM-LOWERING-NEXT:   vector.store %[[SLICE]], %[[TILE_ALLOCA]]
// AFTER-LLVM-LOWERING-NEXT: }
