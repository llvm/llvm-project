// Ensure that omp.simd with the linear clause is translated correctly even
// when otehr loop nests exist in the same function.
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

omp.private {type = private} @test_simd_linear_private_i32 : i32
llvm.func @test_simd_linear() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(2 : i32) : i32
  %3 = llvm.mlir.constant(10 : i32) : i32
  %4 = llvm.alloca %0 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
  %5 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  omp.parallel {
    omp.wsloop private(@test_simd_linear_private_i32 %5 -> %arg0 : !llvm.ptr) {
      omp.loop_nest (%arg1) : i32 = (%1) to (%3) inclusive step (%1) {
        llvm.store %arg1, %arg0 : i32, !llvm.ptr
        llvm.store %2, %4 : i32, !llvm.ptr
        omp.simd linear(%4 : !llvm.ptr = %1 : i32) {
          omp.loop_nest (%arg2) : i32 = (%1) to (%3) inclusive step (%1) {
            llvm.store %arg2, %4 : i32, !llvm.ptr
            omp.yield
          }
        } {linear_var_types = [i32]}
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: void @test_simd_linear()
// CHECK-NOT:     %.linear_result
// CHECK:         %.linear_result = alloca i32
