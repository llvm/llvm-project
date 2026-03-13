// RUN: mlir-opt -split-input-file -verify-diagnostics %s


func.func @missing_permutation(%tc1 : i32, %tc2 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%canonloop2) %iv2 : i32 in range(%tc2) {
      omp.terminator
    }
    omp.terminator
  }

  // expected-error@+1 {{'omp.interchange' op must have permutation attribute}}
  omp.interchange <-(%canonloop1, %canonloop2)

  llvm.return
}

// -----

func.func @no_loops(%tc1 : i32, %tc2 : i32) {
  // expected-error@+1 {{'omp.interchange' op must apply to at least two loops}}
  omp.interchange <-() permutation([2, 1])

  return
}

// -----

func.func @missing_loops(%tc1 : i32, %tc2 : i32) {
  %canonloop1 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc1) {
    omp.terminator
  }

  // expected-error@+1 {{'omp.interchange' op must apply to at least two loops}}
  omp.interchange <-(%canonloop1) permutation([2, 1])

  llvm.return
}

// -----

func.func @wrong_permutation(%tc1 : i32, %tc2 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%canonloop2) %iv2 : i32 in range(%tc2) {
      omp.terminator
    }
    omp.terminator
  }

  // expected-error@+1 {{'omp.interchange' op expecting the same number of permutation attributes and applyees}}
  omp.interchange <-(%canonloop1, %canonloop2) permutation([1 : i32])

  llvm.return
}

// -----

func.func @insufficient_generatees(%tc1 : i32, %tc2 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  %canonloop3 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%canonloop2) %iv2 : i32 in range(%tc2) {
      omp.terminator
    }
    omp.terminator
  }

  // expected-error@+1 {{'omp.interchange' op expecting the same number of generatees and applyees}}
  omp.interchange (%canonloop3) <- (%canonloop1, %canonloop2) permutation([1 : i32, 2 : i32])

  return
}

// -----

func.func @zero_attribute(%tc1 : i32, %tc2 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  %canonloop3 = omp.new_cli
  %canonloop4 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%canonloop2) %iv2 : i32 in range(%tc2) {
      omp.terminator
    }
    omp.terminator
  }

  // expected-error@+1 {{'omp.interchange' op permutation attribute must be a positive integer}}
  omp.interchange (%canonloop3, %canonloop4) <- (%canonloop1, %canonloop2) permutation([0 : i32, 2 : i32])

  return
}

// -----

func.func @zero_attribute(%tc1 : i32, %tc2 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  %canonloop3 = omp.new_cli
  %canonloop4 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%canonloop2) %iv2 : i32 in range(%tc2) {
      omp.terminator
    }
    omp.terminator
  }

  // expected-error@+1 {{'omp.interchange' op every integer from 1 must appear in the permutation attribute}}
  omp.interchange (%canonloop3, %canonloop4) <- (%canonloop1, %canonloop2) permutation([1 : i32, 3 : i32])

  return
}

// -----

func.func @missing_generator(%tc1 : i32, %tc2 : i32) {
  // expected-error@+1 {{'omp.new_cli' op CLI has no generator}}
  %canonloop1 = omp.new_cli

  // expected-note@+1 {{see consumer here: "omp.interchange"(%0) <{operandSegmentSizes = array<i32: 0, 1>, permutation = [1 : i32, 2 : i32]}> : (!omp.cli) -> ()}}
  omp.interchange <-(%canonloop1) permutation([1 : i32, 2 : i32])

  return
}

// -----

func.func @not_perfectly_nested(%tc1 : i32, %tc2 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc1) {
    %v = arith.constant 42 : i32
    omp.canonical_loop(%canonloop2) %iv2 : i32 in range(%tc2) {
      omp.terminator
    }
    omp.terminator
  }

  // expected-error@+1 {{'omp.interchange' op interchanged loop nest must be perfectly nested}}
  omp.interchange <-(%canonloop1, %canonloop2) permutation([1 : i32, 2 : i32])

  llvm.return
}

// -----

func.func @non_nectangular(%tc1 : i32, %tc2 : i32) {
  %canonloop1 = omp.new_cli
  %canonloop2 = omp.new_cli
  omp.canonical_loop(%canonloop1) %iv1 : i32 in range(%tc1) {
    omp.canonical_loop(%canonloop2) %iv2 : i32 in range(%iv1) {
      omp.terminator
    }
    omp.terminator
  }

  // expected-error@+1 {{'omp.interchange' op interchanged loop nest must be rectangular}}
  omp.interchange <-(%canonloop1, %canonloop2) permutation([1 : i32, 2 : i32])

  llvm.return
}
