// RUN: mlir-opt %s -acc-loop-tiling | FileCheck %s

// Test single-level loop tiling with tile(2)
// Original loop: for i = 0 to 10 step 1
// After tiling: tile loop (step=2) containing element loop (step=1)

// CHECK-LABEL: func.func @single_loop_tile
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C10:.*]] = arith.constant 10 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         acc.loop control(%[[IV:.*]] : index) = (%[[C0]] : index) to (%[[C10]] : index) step (%[[C2]] : index) {
// CHECK:           %[[NEW_UB:.*]] = arith.addi %[[IV]], %[[C2]] : index
// CHECK:           %[[MIN_UB:.*]] = arith.minsi %[[NEW_UB]], %[[C10]] : index
// CHECK:           acc.loop control(%[[INNER_IV:.*]] : index) = (%[[IV]] : index) to (%[[MIN_UB]] : index) step (%[[C1]] : index) {
// CHECK:             acc.yield
// CHECK:           } independent
// CHECK:           acc.yield
// CHECK:         } independent
func.func @single_loop_tile(%arg0: memref<10xf32>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  acc.loop tile({%c2 : index}) control(%i : index) = (%c0 : index) to (%c10 : index) step (%c1 : index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i] : memref<10xf32>
    acc.yield
  } independent
  return
}

// Test 2-level nested loop tiling with tile(4, 8)
// Produces one multi-IV tile-group loop (gang) wrapping one multi-IV
// element-group loop (vector), each carrying both tiled induction variables.
// The first tile size applies to the innermost loop: %j is tiled by 4 and %i
// by 8.

// CHECK-LABEL: func.func @nested_loop_tile
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C100:.*]] = arith.constant 100 : index
// CHECK-DAG:     %[[C50:.*]] = arith.constant 50 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index
// Tile-group loop with gang; steps scaled by the tile sizes.
// CHECK:         acc.loop gang control(%[[I:.*]] : index, %[[J:.*]] : index) = (%[[C0]], %[[C0]] : index, index) to (%[[C100]], %[[C50]] : index, index) step (%[[C8]], %[[C4]] : index, index) {
// CHECK:           %[[E0:.*]] = arith.addi %[[I]], %[[C8]] : index
// CHECK:           %[[MUB0:.*]] = arith.minsi %[[E0]], %[[C100]] : index
// CHECK:           %[[E1:.*]] = arith.addi %[[J]], %[[C4]] : index
// CHECK:           %[[MUB1:.*]] = arith.minsi %[[E1]], %[[C50]] : index
// Element-group loop with vector.
// CHECK:           acc.loop vector control(%{{.*}} : index, %{{.*}} : index) = (%[[I]], %[[J]] : index, index) to (%[[MUB0]], %[[MUB1]] : index, index) step (%[[C1]], %[[C1]] : index, index) {
// CHECK:             acc.yield
// CHECK:           } independent
// CHECK:           acc.yield
// CHECK:         } independent
func.func @nested_loop_tile(%arg0: memref<100x50xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  acc.loop gang vector tile({%c4 : index, %c8 : index}) control(%i : index, %j : index) = (%c0, %c0 : index, index) to (%c100, %c50 : index, index) step (%c1, %c1 : index, index) {
    %sum = arith.addi %i, %j : index
    %val = arith.index_castui %sum : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i, %j] : memref<100x50xf32>
    acc.yield
  } independent
  return
}

// Test 3-level loop tiling with tile(2, 4, 8): %k is tiled by 2, %j by 4 and
// %i by 8.

// CHECK-LABEL: func.func @three_dim_tile
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C20:.*]] = arith.constant 20 : index
// CHECK-DAG:     %[[C30:.*]] = arith.constant 30 : index
// CHECK:         acc.loop gang control(%[[I:.*]] : index, %[[J:.*]] : index, %[[K:.*]] : index) = (%[[C0]], %[[C0]], %[[C0]] : index, index, index) to (%[[C10]], %[[C20]], %[[C30]] : index, index, index) step (%[[C8]], %[[C4]], %[[C2]] : index, index, index) {
// CHECK:           %[[E0:.*]] = arith.addi %[[I]], %[[C8]] : index
// CHECK:           %[[MUB0:.*]] = arith.minsi %[[E0]], %[[C10]] : index
// CHECK:           %[[E1:.*]] = arith.addi %[[J]], %[[C4]] : index
// CHECK:           %[[MUB1:.*]] = arith.minsi %[[E1]], %[[C20]] : index
// CHECK:           %[[E2:.*]] = arith.addi %[[K]], %[[C2]] : index
// CHECK:           %[[MUB2:.*]] = arith.minsi %[[E2]], %[[C30]] : index
// CHECK:           acc.loop vector control(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index) = (%[[I]], %[[J]], %[[K]] : index, index, index) to (%[[MUB0]], %[[MUB1]], %[[MUB2]] : index, index, index) step (%[[C1]], %[[C1]], %[[C1]] : index, index, index) {
func.func @three_dim_tile(%arg0: memref<10x20x30xf32>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c30 = arith.constant 30 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 1.0 : f32
  acc.loop gang vector tile({%c2 : index, %c4 : index, %c8 : index}) control(%i : index, %j : index, %k : index) = (%c0, %c0, %c0 : index, index, index) to (%c10, %c20, %c30 : index, index, index) step (%c1, %c1, %c1 : index, index, index) {
    memref.store %cst, %arg0[%i, %j, %k] : memref<10x20x30xf32>
    acc.yield
  } independent
  return
}

// Test tile sizes of different widths: the i32 size tiles the innermost i32
// IV and the i64 size tiles the outermost i64 IV, so no size is wider than
// the IV it tiles.

// CHECK-LABEL: func.func @mixed_width_tile
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i64
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : i32
// CHECK:         acc.loop gang control(%{{.*}} : i64, %{{.*}} : i32) = {{.*}} step (%[[C4]], %[[C8]] : i64, i32) {
// CHECK:           acc.loop vector control(%{{.*}} : i64, %{{.*}} : i32)
func.func @mixed_width_tile(%arg0: memref<100x50xf32>) {
  %c0_i64 = arith.constant 0 : i64
  %c100_i64 = arith.constant 100 : i64
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c0_i32 = arith.constant 0 : i32
  %c50_i32 = arith.constant 50 : i32
  %c1_i32 = arith.constant 1 : i32
  %c8_i32 = arith.constant 8 : i32
  %cst = arith.constant 1.0 : f32
  acc.loop gang vector tile({%c8_i32 : i32, %c4_i64 : i64}) control(%i : i64, %j : i32) = (%c0_i64, %c0_i32 : i64, i32) to (%c100_i64, %c50_i32 : i64, i32) step (%c1_i64, %c1_i32 : i64, i32) {
    %ii = arith.index_cast %i : i64 to index
    %jj = arith.index_cast %j : i32 to index
    memref.store %cst, %arg0[%ii, %jj] : memref<100x50xf32>
    acc.yield
  } independent
  return
}

// Regression test: a loop with GANG(STATIC: N) combined with a multi-dim
// TILE clause used to crash with an assertion failure inside
// LoopOp::getGangValue() when a leftover gang operand was copied onto a
// generated loop without a corresponding gang device-type attribute. Check
// that the pass completes and that gang (with its static operand) is only
// preserved on the tile-group loop, never on the element group.

// CHECK-LABEL: func.func @gang_static_with_multi_dim_tile
// CHECK:         acc.loop gang({static=%{{.*}} : i32}) control(%[[I:.*]] : index, %[[J:.*]] : index) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
// CHECK:           acc.loop control(%{{.*}} : index, %{{.*}} : index) = (%[[I]], %[[J]] : index, index) to ({{.*}}) step ({{.*}}) {
// CHECK-NOT:         gang
// CHECK:             acc.yield
// CHECK:           } independent
// CHECK:           acc.yield
// CHECK:         } independent
func.func @gang_static_with_multi_dim_tile(%arg0: memref<100x50xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %cs = arith.constant 1 : i32
  acc.loop gang({static=%cs : i32}) tile({%c4 : index, %c8 : index}) control(%i : index, %j : index) = (%c0, %c0 : index, index) to (%c100, %c50 : index, index) step (%c1, %c1 : index, index) {
    %sum = arith.addi %i, %j : index
    %val = arith.index_castui %sum : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i, %j] : memref<100x50xf32>
    acc.yield
  } independent
  return
}

// Regression test: a loop with WORKER(N) combined with VECTOR and a
// multi-dim TILE clause used to produce invalid IR because a worker attribute
// was removed from a generated loop without also clearing the worker operand
// value copied onto it, leaving a non-empty worker operand list with no worker
// device-type attribute (rejected by the verifier). Check that the pass
// produces valid IR, with worker only on the tile-group loop and vector only
// on the element-group loop.

// CHECK-LABEL: func.func @worker_num_with_multi_dim_tile
// CHECK:         acc.loop worker(%{{.*}} : i32) control(%[[I:.*]] : index, %[[J:.*]] : index) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
// CHECK-NOT:       vector
// CHECK:           acc.loop vector control(%{{.*}} : index, %{{.*}} : index) = (%[[I]], %[[J]] : index, index) to ({{.*}}) step ({{.*}}) {
// CHECK-NOT:         worker
// CHECK:             acc.yield
// CHECK:           } independent
// CHECK:           acc.yield
// CHECK:         } independent
func.func @worker_num_with_multi_dim_tile(%arg0: memref<100x50xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %cw = arith.constant 4 : i32
  acc.loop worker(%cw : i32) vector tile({%c4 : index, %c8 : index}) control(%i : index, %j : index) = (%c0, %c0 : index, index) to (%c100, %c50 : index, index) step (%c1, %c1 : index, index) {
    %sum = arith.addi %i, %j : index
    %val = arith.index_castui %sum : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i, %j] : memref<100x50xf32>
    acc.yield
  } independent
  return
}

// Test unknown tile size (*) represented as -1
// Should use default tile size (32)

// CHECK-LABEL: func.func @unknown_tile_size
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1000:.*]] = arith.constant 1000 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C32:.*]] = arith.constant 32 : index
// Tile loop with default tile size
// CHECK:         acc.loop control(%[[IV:.*]] : index) = (%[[C0]] : index) to (%[[C1000]] : index) step (%[[C32]] : index) {
// CHECK:           acc.loop control({{.*}} : index) = (%[[IV]] : index) to ({{.*}} : index) step (%[[C1]] : index) {
// CHECK:             acc.yield
// CHECK:           }
// CHECK:           acc.yield
// CHECK:         }
func.func @unknown_tile_size(%arg0: memref<1000xf32>) {
  %c0 = arith.constant 0 : index
  %c1000 = arith.constant 1000 : index
  %c1 = arith.constant 1 : index
  %cm1 = arith.constant -1 : i32  // tile(*) represented as -1
  acc.loop tile({%cm1 : i32}) control(%i : index) = (%c0 : index) to (%c1000 : index) step (%c1 : index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i] : memref<1000xf32>
    acc.yield
  } independent
  return
}

// Test loop with no tile values: pattern should not apply and loop is unchanged.

// CHECK-LABEL: func.func @no_tile_values
// CHECK:         acc.loop control(%{{.*}} : index) = (%{{.*}} : index) to (%{{.*}} : index) step (%{{.*}} : index) {
// CHECK-NOT:       acc.loop
// CHECK:           acc.yield
// CHECK:         }
func.func @no_tile_values() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  acc.loop control(%i : index) = (%c0 : index) to (%c10 : index) step (%c1 : index) {
    acc.yield
  } independent
  return
}

// Test loop tiling when the body contains ops with nested regions.
// Exercises the walk() in moveOpsAndReplaceIVs that must notify the rewriter
// about nested ops (required by MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS).

// CHECK-LABEL: func.func @body_with_nested_region
// CHECK:         acc.loop control(%[[TILE_IV:.*]] : index) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
// CHECK:           acc.loop control(%[[ELEM_IV:.*]] : index) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
// CHECK:             scf.if
// CHECK:               arith.index_castui %[[ELEM_IV]]
// CHECK:             acc.yield
// CHECK:           }
// CHECK:           acc.yield
// CHECK:         }
func.func @body_with_nested_region(%arg0: memref<10xi32>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  acc.loop tile({%c2 : index}) control(%i : index) = (%c0 : index) to (%c10 : index) step (%c1 : index) {
    %threshold = arith.constant 5 : index
    %cond = arith.cmpi ult, %i, %threshold : index
    %val = scf.if %cond -> (i32) {
      %cast = arith.index_castui %i : index to i32
      scf.yield %cast : i32
    } else {
      %c99 = arith.constant 99 : i32
      scf.yield %c99 : i32
    }
    memref.store %val, %arg0[%i] : memref<10xi32>
    acc.yield
  } independent
  return
}

// Test a descending loop: the element bound clamps the tile's far edge up to
// the original bound, so it uses max rather than min.

// CHECK-LABEL: func.func @descending_loop_tile
// CHECK:         %[[CM4:.*]] = arith.constant -4 : index
// CHECK:         acc.loop control(%[[IV:.*]] : index) = (%{{.*}} : index) to (%[[UB:.*]] : index) step (%[[CM4]] : index) {
// CHECK:           %[[EDGE:.*]] = arith.addi %[[IV]], %[[CM4]] : index
// CHECK:           %[[MAX_UB:.*]] = arith.maxsi %[[UB]], %[[EDGE]] : index
// CHECK:           acc.loop control(%{{.*}} : index) = (%[[IV]] : index) to (%[[MAX_UB]] : index) step (%{{.*}} : index) {
func.func @descending_loop_tile(%arg0: memref<16xf32>, %start: index, %floor: index) {
  %cm1 = arith.constant -1 : index
  %c4 = arith.constant 4 : index
  acc.loop tile({%c4 : index}) control(%i : index) = (%start : index) to (%floor : index) step (%cm1 : index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i] : memref<16xf32>
    acc.yield
  } independent
  return
}

// Test a descending loop with an inclusive upper bound: the inclusive edge
// moves in the step's direction, so it adds one instead of subtracting.

// CHECK-LABEL: func.func @descending_inclusive_tile
// CHECK:         %[[CM32:.*]] = arith.constant -32 : index
// CHECK:         acc.loop control(%[[IV:.*]] : index) = (%{{.*}} : index) to (%[[UB:.*]] : index) step (%[[CM32]] : index) {
// CHECK:           %[[STEPPED:.*]] = arith.addi %[[IV]], %[[CM32]] : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[EDGE:.*]] = arith.addi %[[STEPPED]], %[[C1]] : index
// CHECK:           %[[MAX_UB:.*]] = arith.maxsi %[[UB]], %[[EDGE]] : index
// CHECK:           acc.loop control(%{{.*}} : index) = (%[[IV]] : index) to (%[[MAX_UB]] : index) step (%{{.*}} : index) {
func.func @descending_inclusive_tile(%arg0: memref<128xf32>, %start: index, %floor: index) {
  %cm1 = arith.constant -1 : index
  %c32 = arith.constant 32 : index
  acc.loop tile({%c32 : index}) control(%i : index) = (%start : index) to (%floor : index) step (%cm1 : index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i] : memref<128xf32>
    acc.yield
  } inclusiveUpperbound(array<i1: true>) independent
  return
}

// Test an ascending loop with an inclusive upper bound: uses min, and the edge
// subtracts one.

// CHECK-LABEL: func.func @ascending_inclusive_tile
// CHECK:         acc.loop control(%[[IV:.*]] : index) = (%{{.*}} : index) to (%[[UB:.*]] : index) step (%{{.*}} : index) {
// CHECK:           %[[STEPPED:.*]] = arith.addi %[[IV]], %{{.*}} : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[EDGE:.*]] = arith.subi %[[STEPPED]], %[[C1]] : index
// CHECK:           %[[MIN_UB:.*]] = arith.minsi %[[UB]], %[[EDGE]] : index
func.func @ascending_inclusive_tile(%arg0: memref<128xf32>, %lb: index, %ub: index) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  acc.loop tile({%c32 : index}) control(%i : index) = (%lb : index) to (%ub : index) step (%c1 : index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i] : memref<128xf32>
    acc.yield
  } inclusiveUpperbound(array<i1: true>) independent
  return
}

// Test a step whose sign is only known at run time: both clamps are computed
// and selected on the sign.

// CHECK-LABEL: func.func @runtime_step_tile
// CHECK:         acc.loop control(%[[IV:.*]] : index) = (%{{.*}} : index) to (%[[UB:.*]] : index) step (%{{.*}} : index) {
// CHECK:           %[[EDGE:.*]] = arith.addi %[[IV]], %{{.*}} : index
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[IS_DESC:.*]] = arith.cmpi slt, %[[STEP:.*]], %[[ZERO]] : index
// CHECK:           %[[MAX_UB:.*]] = arith.maxsi %[[UB]], %[[EDGE]] : index
// CHECK:           %[[MIN_UB:.*]] = arith.minsi %[[UB]], %[[EDGE]] : index
// CHECK:           %[[SEL:.*]] = arith.select %[[IS_DESC]], %[[MAX_UB]], %[[MIN_UB]] : index
// CHECK:           acc.loop control(%{{.*}} : index) = (%[[IV]] : index) to (%[[SEL]] : index) step (%{{.*}} : index) {
func.func @runtime_step_tile(%arg0: memref<16xf32>, %lb: index, %ub: index, %step: index) {
  %c4 = arith.constant 4 : index
  acc.loop tile({%c4 : index}) control(%i : index) = (%lb : index) to (%ub : index) step (%step : index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i] : memref<16xf32>
    acc.yield
  } independent
  return
}
