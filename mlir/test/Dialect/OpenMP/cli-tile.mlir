// RUN: mlir-opt %s            | FileCheck %s --enable-var-scope
// RUN: mlir-opt %s | mlir-opt | FileCheck %s --enable-var-scope


// Raw syntax check (MLIR output is always pretty-printed)
// CHECK-LABEL: @omp_tile_raw(
// CHECK-SAME: %[[tc:.+]]: i32, %[[ts:.+]]: i32) {
func.func @omp_tile_raw(%tc : i32, %ts : i32) -> () {
  // CHECK-NEXT: %canonloop = omp.new_cli
  %canonloop = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: %grid1 = omp.new_cli
  %grid = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: %intratile1 = omp.new_cli
  %intratile = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  "omp.canonical_loop" (%tc, %canonloop) ({
    ^bb0(%iv: i32):
      // CHECK: omp.terminator
      omp.terminator
  }) : (i32, !omp.cli) -> ()
  // CHECK: omp.tile (%grid1, %intratile1) <- (%canonloop) sizes(%[[ts]] : i32)
  "omp.tile"(%grid,  %intratile, %canonloop, %ts) <{operandSegmentSizes = array<i32: 2, 1, 1>}> : (!omp.cli,  !omp.cli, !omp.cli, i32) -> ()
  //"omp.tile" (%canonloop) : (!omp.cli) -> ()
  return
}


// Pretty syntax check
// CHECK-LABEL: @omp_tile_pretty(
// CHECK-SAME: %[[tc:.+]]: i32, %[[ts:.+]]: i32) {
func.func @omp_tile_pretty(%tc : i32, %ts : i32) -> () {
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %canonloop = omp.new_cli
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %grid = omp.new_cli
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %intratile = omp.new_cli
  // CHECK-NEXT: omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    // CHECK: omp.terminator
    omp.terminator
  }
  // CHECK: omp.tile (%grid1, %intratile1) <- (%canonloop) sizes(%[[ts]] : i32)
  omp.tile(%grid, %intratile) <- (%canonloop) sizes(%ts : i32)
  return
}


// Specifying the generatees for omp.tile is optional
// CHECK-LABEL: @omp_tile_optionalgen_pretty(
// CHECK-SAME: %[[tc:.+]]: i32, %[[ts:.+]]: i32) {
func.func @omp_tile_optionalgen_pretty(%tc : i32, %ts : i32) -> () {
  // CHECK-NEXT: %canonloop = omp.new_cli
  %canonloop = omp.new_cli
  // CHECK-NEXT: omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    // CHECK: omp.terminator
    omp.terminator
  }
  // CHECK: omp.tile <- (%canonloop) sizes(%[[ts]] : i32)
  omp.tile <- (%canonloop) sizes(%ts : i32)
  return
}


// Two-dimensional tiling
// CHECK-LABEL: @omp_tile_2d_pretty(
// CHECK-SAME: %[[tc1:.+]]: i32, %[[tc2:.+]]: i32, %[[ts1:.+]]: i32, %[[ts2:.+]]: i32) {
func.func @omp_tile_2d_pretty(%tc1 : i32, %tc2 : i32, %ts1 : i32, %ts2 : i32) -> () {
  // CHECK-NEXT: %canonloop = omp.new_cli
  %cli_outer = omp.new_cli
  // CHECK-NEXT: %canonloop_d1 = omp.new_cli
  %cli_inner = omp.new_cli
  // CHECK-NEXT: %grid1 = omp.new_cli
  %grid1 = omp.new_cli
  // CHECK-NEXT: %grid2 = omp.new_cli
  %grid2 = omp.new_cli
  // CHECK-NEXT: %intratile1 = omp.new_cli
  %intratile1 = omp.new_cli
  // CHECK-NEXT: %intratile2 = omp.new_cli
  %intratile2 = omp.new_cli
  // CHECK-NEXT:  omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc1]]) {
  omp.canonical_loop(%cli_outer) %iv_outer : i32 in range(%tc1) {
    // CHECK-NEXT: omp.canonical_loop(%canonloop_d1) %iv_d1 : i32 in range(%[[tc2]]) {
    omp.canonical_loop(%cli_inner) %iv_inner : i32 in range(%tc2) {
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  // CHECK:  omp.tile (%grid1, %grid2, %intratile1, %intratile2) <- (%canonloop, %canonloop_d1) sizes(%[[ts1]], %[[ts2]] : i32, i32)
  omp.tile (%grid1, %grid2, %intratile1, %intratile2) <- (%cli_outer, %cli_inner) sizes(%ts1, %ts2 : i32, i32)
  return
}


// Three-dimensional tiling
// CHECK-LABEL: @omp_tile_3d_pretty(
// CHECK-SAME: %[[tc:.+]]: i32, %[[ts:.+]]: i32) {
func.func @omp_tile_3d_pretty(%tc : i32, %ts : i32) -> () {
  // CHECK-NEXT: %canonloop = omp.new_cli
  %cli_outer = omp.new_cli
  // CHECK-NEXT: %canonloop_d1 = omp.new_cli
  %cli_middle = omp.new_cli
  // CHECK-NEXT: %canonloop_d2 = omp.new_cli
  %cli_inner = omp.new_cli
  // CHECK-NEXT: %grid1 = omp.new_cli
  %grid1 = omp.new_cli
  // CHECK-NEXT: %grid2 = omp.new_cli
  %grid2 = omp.new_cli
  // CHECK-NEXT: %grid3 = omp.new_cli
  %grid3 = omp.new_cli
  // CHECK-NEXT: %intratile1 = omp.new_cli
  %intratile1 = omp.new_cli
  // CHECK-NEXT: %intratile2 = omp.new_cli
  %intratile2 = omp.new_cli
  // CHECK-NEXT: %intratile3 = omp.new_cli
  %intratile3 = omp.new_cli
  // CHECK-NEXT:  omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%cli_outer) %iv_outer : i32 in range(%tc) {
    // CHECK-NEXT: omp.canonical_loop(%canonloop_d1) %iv_d1 : i32 in range(%[[tc]]) {
    omp.canonical_loop(%cli_middle) %iv_middle : i32 in range(%tc) {
    // CHECK-NEXT: omp.canonical_loop(%canonloop_d2) %iv_d2 : i32 in range(%[[tc]]) {
      omp.canonical_loop(%cli_inner) %iv_inner : i32 in range(%tc) {
        // CHECK: omp.terminator
        omp.terminator
      }
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  // CHECK:  omp.tile (%grid1, %grid2, %grid3, %intratile1, %intratile2, %intratile3) <- (%canonloop, %canonloop_d1, %canonloop_d2) sizes(%[[ts]], %[[ts]], %[[ts]] : i32, i32, i32)
  omp.tile (%grid1, %grid2, %grid3, %intratile1, %intratile2, %intratile3) <- (%cli_outer, %cli_middle, %cli_inner) sizes(%ts, %ts, %ts: i32, i32, i32)
  return
}


// Composition of multiple tilings
// CHECK-LABEL: @omp_tile_composition(
// CHECK-SAME: %[[tc:.+]]: i32, %[[ts:.+]]: i32, %[[grid_ts:.+]]: i32, %[[intratile_ts:.+]]: i32) {
func.func @omp_tile_composition(%tc: i32, %ts: i32, %grid_ts: i32, %intratile_ts: i32) -> () {
  %canonloop = omp.new_cli
  %grid = omp.new_cli
  %intratile = omp.new_cli
  %grid_intratile = omp.new_cli
  %grid_grid = omp.new_cli
  %intratile_grid = omp.new_cli
  %intratile_intratile = omp.new_cli

  // CHECK: omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.tile (%grid1, %intratile1) <- (%canonloop) sizes(%[[ts]] : i32)
  omp.tile(%grid, %intratile) <- (%canonloop) sizes(%ts : i32)
  // CHECK: omp.tile (%grid1_1, %intratile1_0) <- (%grid1) sizes(%[[grid_ts]] : i32)
  omp.tile(%grid_grid, %grid_intratile) <- (%grid) sizes(%grid_ts : i32)
  // CHECK: omp.tile (%grid1_2, %intratile1_3) <- (%intratile1) sizes(%[[intratile_ts]] : i32)
  omp.tile(%intratile_grid, %intratile_intratile) <- (%intratile) sizes(%intratile_ts : i32)
  // CHECK: return
  return
}


// Composition of multiple 2d-tilings
// CHECK-LABEL: @omp_tile_2d_composition(
// CHECK-SAME: %[[tc1:.+]]: i32, %[[tc2:.+]]: i32, %[[ts:.+]]: i32) {
func.func @omp_tile_2d_composition(%tc1: i32, %tc2: i32, %ts: i32) -> () {
  %cli_outer = omp.new_cli
  %cli_inner = omp.new_cli
  %outer_grid = omp.new_cli
  %inner_grid = omp.new_cli
  %outer_intratile = omp.new_cli
  %inner_intratile = omp.new_cli
  %outer_grid_grid = omp.new_cli
  %inner_grid_grid = omp.new_cli
  %outer_grid_intratile = omp.new_cli
  %inner_grid_intratile = omp.new_cli
  %outer_intratile_grid = omp.new_cli
  %inner_intratile_grid = omp.new_cli
  %outer_intratile_intratile = omp.new_cli
  %inner_intratile_intratile = omp.new_cli

  // CHECK:  omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc1]]) {
  omp.canonical_loop(%cli_outer) %iv_outer : i32 in range(%tc1) {
    // CHECK-NEXT: omp.canonical_loop(%canonloop_d1) %iv_d1 : i32 in range(%[[tc2]]) {
    omp.canonical_loop(%cli_inner) %iv_inner : i32 in range(%tc2) {
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.tile (%grid1, %grid2, %intratile1, %intratile2) <- (%canonloop, %canonloop_d1) sizes(%[[ts]], %[[ts]] : i32, i32)
  omp.tile(%outer_grid, %inner_grid, %outer_intratile, %inner_intratile) <- (%cli_outer, %cli_inner) sizes(%ts, %ts : i32, i32)
  // CHECK: omp.tile (%grid1_0, %grid2_1, %intratile1_2, %intratile2_3) <- (%grid1, %grid2) sizes(%[[ts]], %[[ts]] : i32, i32)
  omp.tile(%outer_grid_grid, %inner_grid_grid, %outer_grid_intratile, %inner_grid_intratile) <- (%outer_grid, %inner_grid) sizes(%ts, %ts : i32, i32)
  // CHECK: omp.tile (%grid1_4, %grid2_5, %intratile1_6, %intratile2_7) <- (%intratile2, %intratile1) sizes(%[[ts]], %[[ts]] : i32, i32)
  omp.tile(%outer_intratile_grid, %inner_intratile_grid, %outer_intratile_intratile, %inner_intratile_intratile) <- (%inner_intratile, %outer_intratile) sizes(%ts, %ts : i32, i32)
  // CHECK: return
  return
}
