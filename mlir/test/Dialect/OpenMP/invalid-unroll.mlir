// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @unroll_full_dynamic_trip_count(%tc : i32) {
  %canonloop = omp.new_cli
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    omp.terminator
  }

  // expected-error@+1 {{'omp.unroll_full' op applyee loop must have a constant trip count}}
  omp.unroll_full(%canonloop)

  return
}

// -----

func.func @unroll_full_constant_trip_count() {
  %tc = arith.constant 100 : i32
  %canonloop = omp.new_cli
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    omp.terminator
  }

  omp.unroll_full(%canonloop)

  return
}

// -----

// unroll_partial has no such requirement.
func.func @unroll_partial_dynamic_trip_count(%tc : i32) {
  %canonloop = omp.new_cli
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    omp.terminator
  }

  omp.unroll_partial(%canonloop) {unroll_factor = 2 : i64}

  return
}
