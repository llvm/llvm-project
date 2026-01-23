// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

omp.private {type = private} @_QFtestEi_private_i32 : i32

omp.private {type = firstprivate} @_QFtestEa_firstprivate_i32 : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}


llvm.func @_QPtest() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(20 : i32) : i32
  llvm.store %6, %3 : i32, !llvm.ptr
  %7 = llvm.mlir.constant(1 : i32) : i32
  %8 = llvm.mlir.constant(5 : i32) : i32
  %9 = llvm.mlir.constant(1 : i32) : i32
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  omp.taskloop priority(%c1_i32 : i32) private(@_QFtestEa_firstprivate_i32 %3 -> %arg0, @_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    omp.loop_nest (%arg2) : i32 = (%7) to (%8) inclusive step (%9) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %10 = llvm.load %arg0 : !llvm.ptr -> i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.add %10, %11 : i32
      llvm.store %12, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK:   %[[omp_global_thread_num:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:   %[[VAL_5:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @1, i32 %[[omp_global_thread_num]], i32 33, i64 40, i64 32, ptr @_QPtest..omp_par)
