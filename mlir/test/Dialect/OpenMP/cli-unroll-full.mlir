// RUN: mlir-opt %s            | FileCheck %s --enable-var-scope
// RUN: mlir-opt %s | mlir-opt | FileCheck %s --enable-var-scope


// CHECK-LABEL: @omp_unroll_full_raw(
// CHECK-SAME: %[[tc:.+]]: i32) {
func.func @omp_unroll_full_raw(%tc : i32) -> () {
  // CHECK-NEXT: %canonloop = omp.new_cli
  %canonloop = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  "omp.canonical_loop" (%tc, %canonloop) ({
    ^bb0(%iv: i32):
      omp.terminator
  }) : (i32, !omp.cli) -> ()
  // CHECK: omp.unroll_full(%canonloop)
  "omp.unroll_full" (%canonloop) : (!omp.cli) -> ()
  return
}


// CHECK-LABEL: @omp_unroll_full_pretty(
// CHECK-SAME: %[[tc:.+]]: i32) {
func.func @omp_unroll_full_pretty(%tc : i32) -> () {
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %canonloop = omp.new_cli
  // CHECK-NEXT:  omp.canonical_loop(%canonloop) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    omp.terminator
  }
  // CHECK: omp.unroll_full(%canonloop)
  omp.unroll_full(%canonloop)
  return
}
