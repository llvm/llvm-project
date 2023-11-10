// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @omp_canonloop_raw
// CHECK-SAME: (%[[tc:.*]]: i32)
func.func @omp_canonloop_raw(%tc : i32) -> () {
  // CHECK: omp.canonical_loop %{{.*}} : i32 in [0, %[[tc]]) {
  "omp.canonical_loop" (%tc) ({
    ^bb0(%iv: i32):
    omp.yield
  }) : (i32) -> ()
  return
}

// CHECK-LABEL: @omp_nested_canonloop_raw
// CHECK-SAME: (%[[tc_outer:.*]]: i32, %[[tc_inner:.*]]: i32)
func.func @omp_nested_canonloop_raw(%tc_outer : i32, %tc_inner : i32) -> () {
  // CHECK: %[[outer_cli:.*]] = omp.new_cli : !omp.cli
  %outer = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK: %[[inner_cli:.*]] = omp.new_cli : !omp.cli
  %inner = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK: omp.canonical_loop %{{.*}} : i32 in [0, %[[tc_outer]]), %[[outer_cli]] : !omp.cli {
  "omp.canonical_loop" (%tc_outer, %outer) ({
    ^bb_outer(%iv_outer: i32):
      // CHECK: omp.canonical_loop %{{.*}} : i32 in [0, %[[tc_inner]]), %[[inner_cli]] : !omp.cli {
      "omp.canonical_loop" (%tc_inner, %inner) ({
        ^bb_inner(%iv_inner: i32):
          omp.yield
      }) : (i32, !omp.cli) -> ()
      omp.yield
  }) : (i32, !omp.cli) -> ()
  return
}

// CHECK-LABEL: @omp_canonloop_pretty
// CHECK-SAME: (%[[tc:.*]]: i32)
func.func @omp_canonloop_pretty(%tc : i32) -> () {
  // CHECK: omp.canonical_loop %[[iv:.*]] : i32 in [0, %[[tc]]) {
  omp.canonical_loop %iv : i32 in [0, %tc) {
    // CHECK-NEXT: %{{.*}} = llvm.add %[[iv]], %[[iv]] : i32
    %newval = llvm.add %iv, %iv: i32
    omp.yield
  }
  return
}

// CHECK-LABEL: @omp_canonloop_nested_pretty
func.func @omp_canonloop_nested_pretty(%tc : i32) -> () {
  // CHECK: %[[cli:.*]] = omp.new_cli : !omp.cli
  %cli = omp.new_cli : !omp.cli
  // CHECK: omp.canonical_loop %{{.*}} : i32 in [0, %{{.*}}), %[[cli]] : !omp.cli {
  omp.canonical_loop %iv1 : i32 in [0, %tc), %cli : !omp.cli {
    // CHECK: omp.canonical_loop %{{.*}} : i32 in [0, %{{.*}}) {
    omp.canonical_loop %iv2 : i32 in [0, %tc) {
      omp.yield
    }
    omp.yield
  }
  return
}

