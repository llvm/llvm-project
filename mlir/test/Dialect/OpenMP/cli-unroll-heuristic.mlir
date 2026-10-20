// RUN: mlir-opt %s            | FileCheck %s --enable-var-scope
// RUN: mlir-opt %s | mlir-opt | FileCheck %s --enable-var-scope


// CHECK-LABEL: @omp_unroll_heuristic_raw(
// CHECK-SAME: %[[tc:.+]]: i32) {
func.func @omp_unroll_heuristic_raw(%tc : i32) -> () {
  // CHECK-NEXT: %canonloop = omp.new_cli
  %canonloop = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  "omp.canonical_loop" (%tc, %canonloop) ({
    ^bb0(%iv: i32):
      omp.terminator
  }) : (i32, !omp.cli) -> ()
  // CHECK: omp.unroll_heuristic(%canonloop)
  "omp.unroll_heuristic" (%canonloop) : (!omp.cli) -> ()
  return
}


// CHECK-LABEL: @omp_unroll_heuristic_pretty(
// CHECK-SAME: %[[tc:.+]]: i32) {
func.func @omp_unroll_heuristic_pretty(%tc : i32) -> () {
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %canonloop = omp.new_cli
  // CHECK-NEXT:  omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    omp.terminator
  }
  // CHECK: omp.unroll_heuristic(%canonloop)
  omp.unroll_heuristic(%canonloop)
  return
}


// CHECK-LABEL: @omp_unroll_heuristic_nested_pretty(
// CHECK-SAME: %[[tc:.+]]: i32) {
func.func @omp_unroll_heuristic_nested_pretty(%tc : i32) -> () {
  // CHECK-NEXT: %canonloop = omp.new_cli
  %cli_outer = omp.new_cli
  // CHECK-NEXT: %canonloop_d1 = omp.new_cli
  %cli_inner = omp.new_cli
  // CHECK-NEXT: omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%cli_outer) %iv_outer : i32 in range(%tc) {
    // CHECK-NEXT: omp.canonical_loop(%canonloop_d1) %iv_d1 : i32 in range(%[[tc]]) {
    omp.canonical_loop(%cli_inner) %iv_inner : i32 in range(%tc) {
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }

  // CHECK: omp.unroll_heuristic(%canonloop)
  omp.unroll_heuristic(%cli_outer)
  // CHECK-NEXT: omp.unroll_heuristic(%canonloop_d1)
  omp.unroll_heuristic(%cli_inner)
  return
}
