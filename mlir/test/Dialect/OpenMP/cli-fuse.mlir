// RUN: mlir-opt %s            | FileCheck %s --enable-var-scope
// RUN: mlir-opt %s | mlir-opt | FileCheck %s --enable-var-scope


// Raw syntax check (MLIR output is always pretty-printed)
// CHECK-LABEL: @omp_fuse_raw(
// CHECK-SAME: %[[tc1:.+]]: i32, %[[tc2:.+]]: i32) {
func.func @omp_fuse_raw(%tc1 : i32, %tc2 : i32) -> () {
  // CHECK-NEXT: %canonloop_s0 = omp.new_cli
  %canonloop_s0 = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: %canonloop_s1 = omp.new_cli
  %canonloop_s1 = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: %fused = omp.new_cli
  %fused = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: omp.canonical_loop(%canonloop_s0) %iv_s0 : i32 in range(%[[tc1]]) {
  "omp.canonical_loop" (%tc1, %canonloop_s0) ({
    ^bb0(%iv_s0: i32):
      // CHECK: omp.terminator
      omp.terminator
  }) : (i32, !omp.cli) -> ()
  // CHECK: omp.canonical_loop(%canonloop_s1) %iv_s1 : i32 in range(%[[tc2]]) {
  "omp.canonical_loop" (%tc2, %canonloop_s1) ({
    ^bb0(%iv_s1: i32):
      // CHECK: omp.terminator
      omp.terminator
  }) : (i32, !omp.cli) -> ()
  // CHECK: omp.fuse (%fused) <- (%canonloop_s0, %canonloop_s1)
  "omp.fuse"(%fused,  %canonloop_s0, %canonloop_s1) <{operandSegmentSizes = array<i32: 1, 2>}> : (!omp.cli,  !omp.cli, !omp.cli) -> ()
  return
}

// Pretty syntax check
// CHECK-LABEL: @omp_fuse_pretty(
// CHECK-SAME: %[[tc1:.+]]: i32, %[[tc2:.+]]: i32) {
func.func @omp_fuse_pretty(%tc1 : i32, %tc2 : i32) -> () {
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %canonloop_s0 = omp.new_cli
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %canonloop_s1 = omp.new_cli
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %fused = omp.new_cli
  // CHECK-NEXT: omp.canonical_loop(%canonloop_s0) %iv_s0 : i32 in range(%[[tc1]]) {
  omp.canonical_loop (%canonloop_s0) %iv_s0 : i32 in range(%tc1) {
      // CHECK: omp.terminator
      omp.terminator
  }
  // CHECK: omp.canonical_loop(%canonloop_s1) %iv_s1 : i32 in range(%[[tc2]]) {
  omp.canonical_loop (%canonloop_s1) %iv_s1 : i32 in range(%tc2) {
      // CHECK: omp.terminator
      omp.terminator
  }
  // CHECK: omp.fuse (%fused) <- (%canonloop_s0, %canonloop_s1)
  omp.fuse(%fused) <- (%canonloop_s0, %canonloop_s1) 
  return
}

// Specifying the generatees for omp.fuse is optional
// CHECK-LABEL: @omp_fuse_optionalgen_pretty(
// CHECK-SAME: %[[tc1:.+]]: i32, %[[tc2:.+]]: i32) {
func.func @omp_fuse_optionalgen_pretty(%tc1 : i32, %tc2 : i32) -> () {
  // CHECK-NEXT: %canonloop_s0 = omp.new_cli
  %canonloop_s0 = omp.new_cli
  // CHECK-NEXT: omp.canonical_loop(%canonloop_s0) %iv_s0 : i32 in range(%[[tc1]]) {
  omp.canonical_loop(%canonloop_s0) %iv_s0 : i32 in range(%tc1) {
    // CHECK: omp.terminator
    omp.terminator
  }
  // CHECK: %canonloop_s1 = omp.new_cli
  %canonloop_s1 = omp.new_cli
  // CHECK-NEXT: omp.canonical_loop(%canonloop_s1) %iv_s1 : i32 in range(%[[tc2]]) {
  omp.canonical_loop(%canonloop_s1) %iv_s1 : i32 in range(%tc2) {
    // CHECK: omp.terminator
    omp.terminator
  }
  // CHECK: omp.fuse <- (%canonloop_s0, %canonloop_s1)
  omp.fuse <- (%canonloop_s0, %canonloop_s1)
  return
}

// Fuse with looprange attributes
// CHECK-LABEL: @omp_fuse_looprange(
// CHECK-SAME: %[[tc1:.+]]: i32, %[[tc2:.+]]: i32, %[[tc3:.+]]: i32) {
func.func @omp_fuse_looprange(%tc1 : i32, %tc2 : i32, %tc3 : i32) -> () {
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %canonloop_s0 = omp.new_cli
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %canonloop_s1 = omp.new_cli
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %canonloop_s2 = omp.new_cli
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %canonloop_fuse = omp.new_cli
  // CHECK-NEXT: %[[CANONLOOP:.+]] = omp.new_cli
  %fused = omp.new_cli
  // CHECK-NEXT: omp.canonical_loop(%canonloop_s0) %iv_s0 : i32 in range(%[[tc1]]) {
  omp.canonical_loop (%canonloop_s0) %iv_s0 : i32 in range(%tc1) {
      // CHECK: omp.terminator
      omp.terminator
  }
  // CHECK: omp.canonical_loop(%canonloop_s1) %iv_s1 : i32 in range(%[[tc2]]) {
  omp.canonical_loop (%canonloop_s1) %iv_s1 : i32 in range(%tc2) {
      // CHECK: omp.terminator
      omp.terminator
  }
  // CHECK: omp.canonical_loop(%canonloop_s2) %iv_s2 : i32 in range(%[[tc3]]) {
  omp.canonical_loop (%canonloop_s2) %iv_s2 : i32 in range(%tc3) {
      // CHECK: omp.terminator
      omp.terminator
  }
  // CHECK: omp.fuse (%canonloop_fuse, %fused) <- (%canonloop_s0,
  // %canonloop_s1, %canonloop_s2) looprange(first = 1, count = 2)
  omp.fuse(%fused, %canonloop_fuse) <- (%canonloop_s0, %canonloop_s1, %canonloop_s2) looprange(first = 1, count = 2)
  return
}

