// Ensure that schedule can be used with the guided kind-type and simd construct
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

omp.private {type = private} @_QFEi_private_i32 : i32
llvm.func @test_simd_guided() {
  %0 = llvm.mlir.constant (1 : i64) : i64
  %c1_i32 = llvm.mlir.constant (1 : i32) : i32
  %c64_i32 = llvm.mlir.constant (64 : i32) : i32
  %c0_i32 = llvm.mlir.constant (0 : i32) : i32
  %c4_i32 = llvm.mlir.constant (4 : i32) : i32
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  omp.wsloop schedule(guided = %c4_i32 : i32, simd) {
    omp.simd linear(%1 = %c1_i32 : !llvm.ptr) private(@_QFEi_private_i32 %1 -> %arg0 : !llvm.ptr) {
      omp.loop_nest (%arg1) : i32 = (%c0_i32) to (%c64_i32) inclusive step (%c1_i32) {
        omp.yield
      }
    } {linear_var_types = [i32], omp.composite}
  } {omp.composite}
  llvm.return
}

// CHECK:   %[[omp_global_thread_num:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:   call void @__kmpc_dispatch_init_4u(ptr @1, i32 %[[omp_global_thread_num]], i32 1073741870, i32 1, i32 65, i32 1, i32 4)
