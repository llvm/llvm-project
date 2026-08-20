// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Check that threadset(omp_pool) sets the free-agent task flag (0x80 = 128) and
// threadset(omp_team) does not.

omp.private {type = private} @_QFtestEi_private_i32 : i32

llvm.func @_QPtest_pool() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %7 = llvm.mlir.constant(1 : i32) : i32
  %8 = llvm.mlir.constant(5 : i32) : i32
  %9 = llvm.mlir.constant(1 : i32) : i32
  omp.taskloop.context threadset(omp_pool) private(@_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%arg2) : i32 = (%7) to (%8) inclusive step (%9) {
        llvm.store %arg2, %arg1 : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  } {omp.combined}
  llvm.return
}

// CHECK-LABEL: define void @_QPtest_pool()
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %{{.+}}, i32 129, i64 {{.+}}, i64 {{.+}}, ptr @_QPtest_pool..omp_par)

// -----

llvm.func @_QPtest_team() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %7 = llvm.mlir.constant(1 : i32) : i32
  %8 = llvm.mlir.constant(5 : i32) : i32
  %9 = llvm.mlir.constant(1 : i32) : i32
  omp.taskloop.context threadset(omp_team) private(@_QFtestEi_private_i32 %1 -> %arg1 : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%arg2) : i32 = (%7) to (%8) inclusive step (%9) {
        llvm.store %arg2, %arg1 : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  } {omp.combined}
  llvm.return
}

// CHECK-LABEL: define void @_QPtest_team()
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %{{.+}}, i32 1, i64 {{.+}}, i64 {{.+}}, ptr @_QPtest_team..omp_par)
