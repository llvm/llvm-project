// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Check that omp.simd as a leaf of a composite construct still generates
// the appropriate loop vectorization attribute.

// CHECK-LABEL: define internal void @test_parallel_do_simd..omp_par
// CHECK: omp.par.entry:
// CHECK: omp.par.region:
// CHECK: omp_loop.header:
// CHECK: omp_loop.inc:
// CHECK-NEXT:   %omp_loop.next = add nuw i32 %omp_loop.iv, 1
// CHECK-NEXT:   br label %omp_loop.header, !llvm.loop ![[LOOP_ATTR:.*]]
// CHECK: ![[LOOP_ATTR]] = distinct !{![[LOOP_ATTR]], ![[LPAR:.*]], ![[LVEC:.*]]}
// CHECK: ![[LPAR]] = !{!"llvm.loop.parallel_accesses", ![[PAR_ACC:.*]]}
// CHECK: ![[LVEC]] = !{!"llvm.loop.vectorize.enable", i1 true}

llvm.func @test_parallel_do_simd() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %2 = llvm.mlir.constant(1000 : i32) : i32
  %3 = llvm.mlir.constant(1 : i32) : i32
  %4 = llvm.mlir.constant(1 : i64) : i64
  omp.parallel {
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.alloca %5 x i32 {bindc_name = "i", pinned} : (i64) -> !llvm.ptr
    %7 = llvm.mlir.constant(1 : i64) : i64
    omp.wsloop {
      omp.simd {
        omp.loop_nest (%arg0) : i32 = (%3) to (%2) inclusive step (%3) {
          llvm.store %arg0, %6 : i32, !llvm.ptr
          omp.yield
        }
      } {omp.composite}
    } {omp.composite}
    omp.terminator
  }
  llvm.return
}
