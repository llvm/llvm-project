// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @omp_canonloop_raw
// CHECK-SAME: (%[[tc:.*]]: i32)
func.func @omp_canonloop_raw(%tc : i32) -> () {
  // CHECK: %{{.*}} = omp.canonical_loop %{{.*}} : i32 in [0, %[[tc]]) {
  %cli = "omp.canonical_loop" (%tc) ({
    ^bb0(%iv: i32):
      // omp.yield without argument is implicit
      // CHECK-NOT: omp.yield
      omp.yield
  }) : (i32) -> (!omp.cli)
  return
}

// CHECK-LABEL: @omp_nested_canonloop_raw
// CHECK-SAME: (%[[tc_outer:.*]]: i32, %[[tc_inner:.*]]: i32)
func.func @omp_nested_canonloop_raw(%tc_outer : i32, %tc_inner : i32) -> () {
  // CHECK: %{{.*}} = omp.canonical_loop %{{.*}} : i32 in [0, %[[tc_outer]]) {
  %outer,%inner = "omp.canonical_loop" (%tc_outer) ({
    ^bb_outer(%iv_outer: i32):
      // CHECK: %[[inner_cli:.*]] = omp.canonical_loop %{{.*}} : i32 in [0, %[[tc_inner]]) {
      %inner = "omp.canonical_loop" (%tc_inner) ({
        ^bb_inner(%iv_inner: i32):
          omp.yield
      }) : (i32) -> (!omp.cli)
      // CHECK: omp.yield(%[[inner_cli]] : !omp.cli)
      omp.yield (%inner : !omp.cli)
  }) : (i32) -> (!omp.cli, !omp.cli)
  return
}

// CHECK-LABEL: @omp_triple_nested_canonloop_raw
func.func @omp_triple_nested_canonloop_raw(%tc_outer : i32,%tc_middle : i32, %tc_inner : i32) -> () {
  // CHECK: %{{.*}} = omp.canonical_loop %{{.*}} : i32 in [0, %{{.*}}) {
  %outer, %middle, %inner = "omp.canonical_loop" (%tc_outer) ({
    ^bb_outer(%iv_outer: i32):
      // CHECK: %[[middle:.*]]:2 = omp.canonical_loop %{{.*}} : i32 in [0, %{{.*}}) {
      %middle, %inner= "omp.canonical_loop" (%tc_middle) ({
        ^bb_middle(%iv_middle: i32):
          // CHECK: %[[inner:.*]] = omp.canonical_loop %{{.*}} : i32 in [0, %{{.*}}) {
          %inner = "omp.canonical_loop" (%tc_inner) ({
            ^bb_inner(%iv_inner: i32):
              omp.yield
          }) : (i32) -> (!omp.cli)
        // CHECK: omp.yield(%[[inner]] : !omp.cli)
        omp.yield (%inner : !omp.cli)
    }) : (i32) -> (!omp.cli,!omp.cli)
    // CHECK: omp.yield(%[[middle]]#0, %[[middle]]#1 : !omp.cli, !omp.cli)
    omp.yield (%middle, %inner : !omp.cli, !omp.cli)
  }) : (i32) -> (!omp.cli, !omp.cli, !omp.cli)
  return
}

// CHECK-LABEL: @omp_canonloop_pretty
// CHECK-SAME: (%[[tc:.*]]: i32)
func.func @omp_canonloop_pretty(%tc : i32) -> () {
  // CHECK: %{{.*}} = omp.canonical_loop %[[iv:.*]] : i32 in [0, %[[tc]]) {
  %cli = omp.canonical_loop %iv : i32 in [0, %tc) {
    // CHECK-NEXT: %{{.*}} = llvm.add %[[iv]], %[[iv]] : i32
    %newval = llvm.add %iv, %iv: i32
    // CHECK-NOT: omp.yield
  }
  return
}

// CHECK-LABEL: @omp_canonloop_implicit_yield
func.func @omp_canonloop_implicit_yield(%tc : i32) -> () {
  // CHECK: %{{.*}} = omp.canonical_loop %{{.*}} : i32 in [0, %{{.*}}) {
  %cli = omp.canonical_loop %iv : i32 in [0, %tc) {
    // CHECK-NOT: omp.yield
    // CHECK-NEXT: }
  }
  return
}

// CHECK-LABEL: @omp_canonloop_nested_pretty
func.func @omp_canonloop_nested_pretty(%tc : i32) -> () {
  // CHECK: %{{.*}} = omp.canonical_loop %{{.*}} : i32 in [0, %{{.*}}) {
  %outer,%inner = omp.canonical_loop %iv1 : i32 in [0, %tc) {
    // CHECK: %[[inner:.*]] = omp.canonical_loop %{{.*}} : i32 in [0, %{{.*}}) {
    %inner = omp.canonical_loop %iv2 : i32 in [0, %tc) {}
    // CHECK:  omp.yield(%[[inner]] : !omp.cli)
    omp.yield (%inner : !omp.cli)
  }
  return
}

