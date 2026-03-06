// RUN: mlir-opt %s            | FileCheck %s --enable-var-scope
// RUN: mlir-opt %s | mlir-opt | FileCheck %s --enable-var-scope


// Raw syntax check (MLIR output is always pretty-printed)
// CHECK-LABEL: @omp_interchange_raw(
// CHECK-SAME: %[[tc1:.+]]: i32, %[[tc2:.+]]: i32) {
func.func @omp_interchange_raw(%tc1 : i32, %tc2 : i32) -> () {
  // CHECK-NEXT: %canonloop = omp.new_cli
  %cli_outer = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: %canonloop_d1 = omp.new_cli
  %cli_inner = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: %interchange = omp.new_cli
  %interchange1 = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: %interchange_0 = omp.new_cli
  %interchange2 = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc1]]) {
  "omp.canonical_loop" (%tc1, %cli_outer) ({
    ^bb0(%iv: i32):
  // CHECK-NEXT: omp.canonical_loop(%canonloop_d1) %iv_d1 : i32 in range(%[[tc2]]) {
    "omp.canonical_loop" (%tc2, %cli_inner) ({
      ^bb0(%iv_d1: i32):
        // CHECK: omp.terminator
        omp.terminator
    }) : (i32, !omp.cli) -> ()
      // CHECK: omp.terminator
      omp.terminator
  }) : (i32, !omp.cli) -> ()
  // CHECK: omp.interchange (%interchange, %interchange_0) <- (%canonloop, %canonloop_d1)
  // CHECK-SAME: permutation([2 : i32, 1 : i32])
  "omp.interchange" (%interchange1, %interchange2, %cli_outer, %cli_inner) <{operandSegmentSizes = array<i32: 2, 2>, permutation = [2 : i32, 1 : i32]}> : (!omp.cli, !omp.cli, !omp.cli, !omp.cli) -> ()
  return
}


// Pretty syntax check
// CHECK-LABEL: @omp_interchange_pretty(
// CHECK-SAME: %[[tc1:.+]]: i32, %[[tc2:.+]]: i32) {
func.func @omp_interchange_pretty(%tc1 : i32, %tc2 : i32) -> () {
  // CHECK-NEXT: %canonloop = omp.new_cli
  %cli_outer = omp.new_cli
  // CHECK-NEXT: %canonloop_d1 = omp.new_cli
  %cli_inner = omp.new_cli
  // CHECK-NEXT: %interchange = omp.new_cli
  %interchange1 = omp.new_cli
  // CHECK-NEXT: %interchange_0 = omp.new_cli
  %interchange2 = omp.new_cli
  // CHECK-NEXT: omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc1]]) {
  omp.canonical_loop(%cli_outer) %iv_outer : i32 in range(%tc1) {
    // CHECK-NEXT: omp.canonical_loop(%canonloop_d1) %iv_d1 : i32 in range(%[[tc2]]) {
    omp.canonical_loop(%cli_inner) %iv_inner : i32 in range(%tc2) {
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  // CHECK: omp.interchange (%interchange, %interchange_0) <- (%canonloop, %canonloop_d1)
  // CHECK-SAME: permutation([2 : i32, 1 : i32])
  omp.interchange (%interchange1, %interchange2) <- (%cli_outer, %cli_inner) permutation([2 : i32, 1 : i32])
  return
}

