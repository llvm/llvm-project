// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s


// CHECK-LABEL: @omp_canonloop_raw(
// CHECK-SAME: %[[tc:.+]]: i32)
func.func @omp_canonloop_raw(%tc : i32) -> () {
  // CHECK: omp.canonical_loop %iv : i32 in range(%[[tc]]) {
  "omp.canonical_loop" (%tc) ({
    ^bb0(%iv: i32):
      // CHECK-NEXT: = llvm.add %iv, %iv : i32
      %newval = llvm.add %iv, %iv : i32
      // CHECK-NEXT: omp.terminator
      omp.terminator
  // CHECK-NEXT: }
  }) : (i32) -> ()
  // CHECK-NEXT: return
  return
}


// CHECK-LABEL: @omp_canonloop_sequential_raw(
// CHECK-SAME: %[[tc:.+]]: i32)
func.func @omp_canonloop_sequential_raw(%tc : i32) -> () {
  // CHECK-NEXT: %canonloop_s0 = omp.new_cli
  %canonloop_s0 = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT:  omp.canonical_loop(%canonloop_s0) %iv : i32 in range(%[[tc]]) {
  "omp.canonical_loop" (%tc, %canonloop_s0) ({
    ^bb_first(%iv_first: i32):
      // CHECK-NEXT: = llvm.add %iv, %iv : i32
      %newval = llvm.add %iv_first, %iv_first : i32
    // CHECK-NEXT: omp.terminator
    omp.terminator
  // CHECK-NEXT: }
  }) : (i32, !omp.cli) -> ()

  // CHECK-NEXT: %canonloop_s1 = omp.new_cli
  %canonloop_s1 = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: omp.canonical_loop(%canonloop_s1) %iv : i32 in range(%[[tc]]) {
  "omp.canonical_loop" (%tc, %canonloop_s1) ({
    ^bb_second(%iv_second: i32):
    // CHECK: omp.terminator
    omp.terminator
  // CHECK-NEXT: }
  }) : (i32, !omp.cli) -> ()

  // CHECK-NEXT: return
  return
}


// CHECK-LABEL: @omp_nested_canonloop_raw(
// CHECK-SAME: %[[tc_outer:.+]]: i32, %[[tc_inner:.+]]: i32)
func.func @omp_nested_canonloop_raw(%tc_outer : i32, %tc_inner : i32) -> () {
  // CHECK-NEXT: %canonloop_s0 = omp.new_cli
  %outer = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: %canonloop_s0_s0 = omp.new_cli
  %inner = "omp.new_cli" () : () -> (!omp.cli)
  // CHECK-NEXT: omp.canonical_loop(%canonloop_s0) %iv : i32 in range(%[[tc_outer]]) {
  "omp.canonical_loop" (%tc_outer, %outer) ({
    ^bb_outer(%iv_outer: i32):
      // CHECK-NEXT: omp.canonical_loop(%canonloop_s0_s0) %iv_0 : i32 in range(%[[tc_inner]]) {
      "omp.canonical_loop" (%tc_inner, %inner) ({
        ^bb_inner(%iv_inner: i32):
          // CHECK-NEXT: = llvm.add %iv, %iv_0 : i32
          %newval = llvm.add %iv_outer, %iv_inner: i32
          // CHECK-NEXT: omp.terminator
          omp.terminator
      }) : (i32, !omp.cli) -> ()
      // CHECK: omp.terminator
      omp.terminator
  }) : (i32, !omp.cli) -> ()
  return
}


// CHECK-LABEL: @omp_canonloop_pretty(
// CHECK-SAME: %[[tc:.+]]: i32)
func.func @omp_canonloop_pretty(%tc : i32) -> () {
  // CHECK-NEXT: omp.canonical_loop %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop %iv : i32 in range(%tc) {
    // CHECK-NEXT: llvm.add %iv, %iv : i32
    %newval = llvm.add %iv, %iv: i32
    // CHECK-NEXT: omp.terminator
    omp.terminator
  }
  return
}


// CHECK-LABEL: @omp_canonloop_constant_pretty()
func.func @omp_canonloop_constant_pretty() -> () {
  // CHECK-NEXT:  %[[tc:.+]] = llvm.mlir.constant(4 : i32) : i32
  %tc = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: omp.canonical_loop %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop %iv : i32 in range(%tc) {
    // CHECK-NEXT: llvm.add %iv, %iv : i32
    %newval = llvm.add %iv, %iv: i32
    // CHECK-NEXT: omp.terminator
    omp.terminator
  }
  return
}


// CHECK-LABEL: @omp_canonloop_sequential_pretty(
// CHECK-SAME: %[[tc:.+]]: i32)
func.func @omp_canonloop_sequential_pretty(%tc : i32) -> () {
  // CHECK-NEXT: %canonloop_s0 = omp.new_cli
  %canonloop_s0 = omp.new_cli
  // CHECK-NEXT:  omp.canonical_loop(%canonloop_s0) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%canonloop_s0) %iv : i32 in range(%tc) {
    // CHECK-NEXT: omp.terminator
    omp.terminator
  }

  // CHECK: %canonloop_s1 = omp.new_cli
  %canonloop_s1 = omp.new_cli
  // CHECK-NEXT:  omp.canonical_loop(%canonloop_s1) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%canonloop_s1) %iv_0 : i32 in range(%tc) {
    // CHECK-NEXT: omp.terminator
    omp.terminator
  }

  return
}


// CHECK-LABEL: @omp_canonloop_nested_pretty(
// CHECK-SAME: %[[tc:.+]]: i32)
func.func @omp_canonloop_nested_pretty(%tc : i32) -> () {
  // CHECK-NEXT: %canonloop_s0 = omp.new_cli
  %canonloop_s0 = omp.new_cli
  // CHECK-NEXT: %canonloop_s0_s0 = omp.new_cli
  %canonloop_s0_s0 = omp.new_cli
  // CHECK-NEXT:  omp.canonical_loop(%canonloop_s0) %iv : i32 in range(%[[tc]]) {
  omp.canonical_loop(%canonloop_s0) %iv : i32 in range(%tc) {
    // CHECK-NEXT: omp.canonical_loop(%canonloop_s0_s0) %iv_0 : i32 in range(%[[tc]]) {
    omp.canonical_loop(%canonloop_s0_s0) %iv_0 : i32 in range(%tc) {
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: omp.terminator
    omp.terminator
  }
  return
}


// CHECK-LABEL: @omp_newcli_unused(
// CHECK-SAME: )
func.func @omp_newcli_unused() -> () {
  // CHECK-NEXT:  %cli = omp.new_cli
  %cli = omp.new_cli
  // CHECK-NEXT: return
  return
}
