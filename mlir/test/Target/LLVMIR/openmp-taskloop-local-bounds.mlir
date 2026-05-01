// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Taskloop loop bounds defined inside omp.taskloop.context by pure operations
// that do not depend on block arguments should be materialized in the parent
// function before the runtime call and translated again inside the outlined
// task.

omp.private {type = private} @_QPtest_taskloop_local_bounds_private_i32 : i32

// CHECK-LABEL: define void @_QPtest_taskloop_local_constants(
llvm.func @_QPtest_taskloop_local_constants() {
  %one_i64 = llvm.mlir.constant(1 : i64) : i64
  %i = llvm.alloca %one_i64 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  omp.taskloop.context private(@_QPtest_taskloop_local_bounds_private_i32 %i -> %arg0 : !llvm.ptr) {
    %lb = llvm.mlir.constant(1 : i32) : i32
    %ub = llvm.mlir.constant(10 : i32) : i32
    %step = llvm.mlir.constant(1 : i32) : i32
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        llvm.store %iv, %arg0 : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK:         %[[GEP_LB:.*]] = getelementptr
// CHECK:         store i64 1, ptr %[[GEP_LB]]
// CHECK:         %[[GEP_UB:.*]] = getelementptr
// CHECK:         store i64 10, ptr %[[GEP_UB]]
// CHECK:         %[[GEP_STEP:.*]] = getelementptr
// CHECK:         store i64 1, ptr %[[GEP_STEP]]
// CHECK:         call void @__kmpc_taskloop(

// CHECK-LABEL: define internal void @_QPtest_taskloop_local_constants..omp_par(
// CHECK:         %[[OL_GEP_LB:.*]] = getelementptr
// CHECK:         %[[OL_LOAD_LB:.*]] = load i64, ptr %[[OL_GEP_LB]]
// CHECK:         %[[OL_GEP_UB:.*]] = getelementptr
// CHECK:         %[[OL_LOAD_UB:.*]] = load i64, ptr %[[OL_GEP_UB]]

// CHECK-LABEL: define void @_QPtest_taskloop_local_derived_bound(
llvm.func @_QPtest_taskloop_local_derived_bound() {
  %one_i64 = llvm.mlir.constant(1 : i64) : i64
  %i = llvm.alloca %one_i64 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  omp.taskloop.context private(@_QPtest_taskloop_local_bounds_private_i32 %i -> %arg0 : !llvm.ptr) {
    %lb = llvm.mlir.constant(1 : i32) : i32
    %ten = llvm.mlir.constant(10 : i32) : i32
    %two = llvm.mlir.constant(2 : i32) : i32
    %ub = llvm.add %ten, %two : i32
    %step = llvm.mlir.constant(1 : i32) : i32
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
        llvm.store %iv, %arg0 : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK:         store i64 12, ptr %{{.*}}
// CHECK:         call void @__kmpc_taskloop(
