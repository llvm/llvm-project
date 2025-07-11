// RUN: mlir-opt %s -verify-diagnostics


func.func @omp_canonloop_block_arg(%tc : i32) -> () {
  // expected-error@below {{Canonical loop region must have exactly one argument}}
  "omp.canonical_loop" (%tc) ({
    ^bb0(%iv: i32, %somearg: i32):
      omp.terminator
  }) : (i32) -> ()
  return
}


func.func @omp_canonloop_multiple_generators(%tc : i32) -> () {
  // expected-error@below {{'omp.new_cli' op CLI must have at most one generator}}
  %cli = omp.new_cli
  // expected-note@below {{second generator here}}
  omp.canonical_loop(%cli) %iv1 : i32 in range(%tc) {
    omp.terminator
  }
  // expected-note@below {{first generator here}}
  omp.canonical_loop(%cli) %iv2 : i32 in range(%tc) {
    omp.terminator
  }
  return
}


func.func @omp_canonloop_multiple_consumers() -> () {
  // expected-error@below {{'omp.new_cli' op CLI must have at most one consumer}}
  %cli = omp.new_cli
  %tc = llvm.mlir.constant(4 : i32) : i32
  omp.canonical_loop(%cli) %iv1 : i32 in range(%tc) {
    omp.terminator
  }
  // expected-note@below {{second consumer here}}
  omp.unroll_heuristic(%cli)
  // expected-note@below {{first consumer here}}
  omp.unroll_heuristic(%cli)
  return
}


func.func @omp_canonloop_no_generator() -> () {
  // expected-error@below {{'omp.new_cli' op CLI has no generator}}
  %cli = omp.new_cli
  // expected-note@below {{see consumer here}}
  omp.unroll_heuristic(%cli)
  return
}
