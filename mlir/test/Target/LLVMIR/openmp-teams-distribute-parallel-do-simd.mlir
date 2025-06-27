// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// Check that omp.simd as a leaf of a composite construct still generates
// the appropriate loop vectorization attribute.

// CHECK-LABEL: define internal void @test_teams_distribute_parallel_do_simd..omp_par
// CHECK: ![[VAL:.*]] = !{!"llvm.loop.vectorize.enable", i1 true}

omp.private {type = private} @_QFEi_private_i32 : i32
llvm.func @test_teams_distribute_parallel_do_simd() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %2 = llvm.mlir.constant(1000 : i32) : i32
  %3 = llvm.mlir.constant(1 : i32) : i32
  %4 = llvm.mlir.constant(1 : i64) : i64
  omp.teams {
    omp.parallel {
      omp.distribute {
        omp.wsloop {
          omp.simd private(@_QFEi_private_i32 %1 -> %arg0 : !llvm.ptr) {
            omp.loop_nest (%arg1) : i32 = (%3) to (%2) inclusive step (%3) {
              llvm.store %arg1, %arg0 : i32, !llvm.ptr
              omp.yield
            }
          } {omp.composite}
        } {omp.composite}
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    omp.terminator
  }
  llvm.return
}
