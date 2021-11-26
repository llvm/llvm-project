// RUN: not mlir-translate -mlir-to-llvmir -split-input-file %s 2>&1 | FileCheck %s

llvm.func @test_omp_wsloop_dynamic_bad_modifier(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(dynamic, ginandtonic) {
    // CHECK: unknown modifier type: ginandtonic
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_many_modifier(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(dynamic, monotonic, monotonic, monotonic) {
    // CHECK: unexpected modifier(s)
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_wrong_modifier(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(dynamic, simd, monotonic) {
    // CHECK: incorrect modifier order
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_wrong_modifier2(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(dynamic, monotonic, monotonic) {
    // CHECK: incorrect modifier order
    omp.yield
  }
  llvm.return
}

// -----

llvm.func @test_omp_wsloop_dynamic_wrong_modifier3(%lb : i64, %ub : i64, %step : i64) -> () {
  omp.wsloop (%iv) : i64 = (%lb) to (%ub) step (%step) schedule(dynamic, simd, simd) {
    // CHECK: incorrect modifier order
    omp.yield
  }
  llvm.return
}
