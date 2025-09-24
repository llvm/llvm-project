// RUN: mlir-opt -split-input-file -verify-diagnostics %s


func.func @missing_sizes(%tc : i32, %ts : i32) {
  %canonloop = omp.new_cli
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    omp.terminator
  }

  // expected-error@+1 {{'omp.tile' op there must be one tile size for each applyee}}
  omp.tile <-(%canonloop)

  llvm.return
}

// -----

func.func @no_loop(%tc : i32, %ts : i32) {
  // expected-error@+1 {{'omp.tile' op must apply to at least one loop}}
  omp.tile <-()

  return
}

// -----

func.func @missing_generator(%tc : i32, %ts : i32) {
  // expected-error@+1 {{'omp.new_cli' op CLI has no generator}}
  %canonloop = omp.new_cli

  // expected-note@+1 {{see consumer here: "omp.tile"(%0, %arg1) <{operandSegmentSizes = array<i32: 0, 1, 1>}> : (!omp.cli, i32) -> ()}}
  omp.tile <-(%canonloop) sizes(%ts : i32)

  return
}

// -----

func.func @insufficient_sizes(%tc : i32, %ts : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv : i32 in range(%tc) {
    omp.terminator
  }
  omp.canonical_loop(%canonloop2) %iv : i32 in range(%tc) {
    omp.terminator
  }

  // expected-error@+1 {{'omp.tile' op there must be one tile size for each applyee}}
  omp.tile <-(%canonloop1, %canonloop2) sizes(%ts : i32)

  llvm.return
}

// -----

func.func @insufficient_applyees(%tc : i32, %ts : i32) {
  %canonloop = omp.new_cli
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    omp.terminator
  }

  // expected-error@+1 {{omp.tile' op there must be one tile size for each applyee}}
  omp.tile <- (%canonloop) sizes(%ts, %ts : i32, i32)

  return
}

// -----

func.func @insufficient_generatees(%tc : i32, %ts : i32) {
  %canonloop = omp.new_cli
  %grid = omp.new_cli
  omp.canonical_loop(%canonloop) %iv : i32 in range(%tc) {
    omp.terminator
  }

  // expected-error@+1 {{'omp.tile' op expecting two times the number of generatees than applyees}}
  omp.tile (%grid) <- (%canonloop) sizes(%ts : i32)

  return
}

// -----

func.func @not_perfectly_nested(%tc : i32, %ts : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc) {
    %v = arith.constant 42 : i32
    omp.canonical_loop(%canonloop2) %iv2 : i32 in range(%tc) {
      omp.terminator
    }
    omp.terminator
  }

  // expected-error@+1 {{'omp.tile' op tiled loop nest must be perfectly nested}}
  omp.tile <-(%canonloop1, %canonloop2) sizes(%ts, %ts : i32, i32)

  llvm.return
}

// -----

func.func @non_nectangular(%tc : i32, %ts : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc) {
    omp.canonical_loop(%canonloop2) %iv2 : i32 in range(%iv1) {
      omp.terminator
    }
    omp.terminator
  }

  // expected-error@+1 {{'omp.tile' op tiled loop nest must be rectangular}}
  omp.tile <-(%canonloop1, %canonloop2) sizes(%ts, %ts : i32, i32)

  llvm.return
}
