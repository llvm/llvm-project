// RUN: mlir-opt -split-input-file -verify-diagnostics %s


func.func @no_loops(%tc1 : i32, %tc2 : i32) {
  // expected-error@+1 {{'omp.fuse' op must apply to at least two loops}}
  omp.fuse <-()

  return
}

// -----

func.func @one_loop(%tc1 : i32, %tc2 : i32) {
  %canonloop = omp.new_cli
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc1) {
    omp.terminator
  }
  // expected-error@+1 {{'omp.fuse' op must apply to at least two loops}}
  omp.fuse <-(%canonloop)

  return
}

// -----

func.func @missing_generator(%tc1 : i32, %tc2 : i32) {
  // expected-error@+1 {{'omp.new_cli' op CLI has no generator}}
  %canonloop = omp.new_cli

  // expected-note@+1 {{see consumer here: "omp.fuse"(%0) <{operandSegmentSizes = array<i32: 0, 1>}> : (!omp.cli) -> ()}}
  omp.fuse <-(%canonloop)

  return
}

// -----

func.func @wrong_generatees1(%tc1 : i32, %tc2 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv : i32 in range(%tc1) {
    omp.terminator
  }
  omp.canonical_loop(%canonloop2) %iv : i32 in range(%tc2) {
    omp.terminator
  }

  %fused1 = omp.new_cli
  %fused2 = omp.new_cli
  // expected-error@+1 {{'omp.fuse' op in a complete fuse the number of generatees must be exactly 1}}
  omp.fuse (%fused1, %fused2) <-(%canonloop1, %canonloop2) 

  llvm.return
}

// -----

func.func @wrong_generatees2(%tc1 : i32, %tc2 : i32, %tc3 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  %canonloop3 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv : i32 in range(%tc1) {
    omp.terminator
  }
  omp.canonical_loop(%canonloop2) %iv : i32 in range(%tc2) {
    omp.terminator
  }
  omp.canonical_loop(%canonloop3) %iv : i32 in range(%tc3) {
    omp.terminator
  }

  %fused = omp.new_cli
  // expected-error@+1 {{'omp.fuse' op the number of generatees must be the number of aplyees plus one minus count}} 
  omp.fuse (%fused) <-(%canonloop1, %canonloop2, %canonloop3) looprange(first = 1, count = 2)

  llvm.return
}

// -----

func.func @wrong_applyees(%tc1 : i32, %tc2 : i32, %tc3 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  %canonloop3 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv : i32 in range(%tc1) {
    omp.terminator
  }
  omp.canonical_loop(%canonloop2) %iv : i32 in range(%tc2) {
    omp.terminator
  }
  omp.canonical_loop(%canonloop3) %iv : i32 in range(%tc3) {
    omp.terminator
  }

  %fused = omp.new_cli
  %canonloop_fuse = omp.new_cli
  // expected-error@+1 {{'omp.fuse' op the numbers of applyees must be at least first minus one plus count attributes}}
  omp.fuse (%fused, %canonloop_fuse) <-(%canonloop1, %canonloop2, %canonloop3) looprange(first = 1, count = 5)

  llvm.return
}

