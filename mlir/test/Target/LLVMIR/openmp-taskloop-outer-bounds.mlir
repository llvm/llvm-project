// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Regression test: taskloop loop bounds defined outside omp.taskloop must not
// leave the outlined loop body using the original function's casted step
// value. The outlined loop preheader has to reload the task's step from the
// task shared values structure.

omp.private {type = private} @_QFtestEi_private_i32 : i32

llvm.func @_QPtest() {
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %i = llvm.alloca %c1_i64 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %lb.addr = llvm.alloca %c1_i64 x i32 {bindc_name = "lb"} : (i64) -> !llvm.ptr
  %ub.addr = llvm.alloca %c1_i64 x i32 {bindc_name = "ub"} : (i64) -> !llvm.ptr
  %step.addr = llvm.alloca %c1_i64 x i32 {bindc_name = "step"} : (i64) -> !llvm.ptr
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c10_i32 = llvm.mlir.constant(10 : i32) : i32
  %c2_i32 = llvm.mlir.constant(2 : i32) : i32
  llvm.store %c1_i32, %lb.addr : i32, !llvm.ptr
  llvm.store %c10_i32, %ub.addr : i32, !llvm.ptr
  llvm.store %c2_i32, %step.addr : i32, !llvm.ptr
  %lb = llvm.load %lb.addr : !llvm.ptr -> i32
  %ub = llvm.load %ub.addr : !llvm.ptr -> i32
  %step = llvm.load %step.addr : !llvm.ptr -> i32
  omp.taskloop private(@_QFtestEi_private_i32 %i -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) inclusive step (%step) {
      llvm.store %iv, %arg0 : i32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @_QPtest() {
// CHECK: call void @__kmpc_taskloop(

// CHECK-LABEL: define internal void @_QPtest..omp_par(
// CHECK: %[[GEPSTEP:.*]] = getelementptr {{.*}}, ptr %{{.*}}, i32 0, i32 2
// CHECK: %[[LOADSTEP:.*]] = load i64, ptr %[[GEPSTEP]]
// CHECK: %[[SUB:.*]] = sub i64 %{{.*}}, %{{.*}}
// CHECK: %[[DIV:.*]] = sdiv i64 %[[SUB]], %[[LOADSTEP]]
