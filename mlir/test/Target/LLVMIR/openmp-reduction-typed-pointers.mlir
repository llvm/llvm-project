// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Only check the overall shape of the code and the presence of relevant
// runtime calls. Actual IR checking is done at the OpenMPIRBuilder level.

omp.reduction.declare @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = llvm.fadd %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr<f32>, %arg3: !llvm.ptr<f32>):
  %2 = llvm.load %arg3 : !llvm.ptr<f32>
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr<f32>, f32
  omp.yield
}

// CHECK-LABEL: @simple_reduction
llvm.func @simple_reduction(%lb : i64, %ub : i64, %step : i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr<f32>
  omp.parallel {
    omp.wsloop reduction(@add_f32 -> %0 : !llvm.ptr<f32>)
    for (%iv) : i64 = (%lb) to (%ub) step (%step) {
      %1 = llvm.mlir.constant(2.0 : f32) : f32
      omp.reduction %1, %0 : f32, !llvm.ptr<f32>
      omp.yield
    }
    omp.terminator
  }
  llvm.return
}

// Call to the outlined function.
// CHECK: call void {{.*}} @__kmpc_fork_call
// CHECK-SAME: @[[OUTLINED:[A-Za-z_.][A-Za-z0-9_.]*]]

// Outlined function.
// CHECK: define internal void @[[OUTLINED]]

// Private reduction variable and its initialization.
// CHECK: %[[PRIVATE:.+]] = alloca float
// CHECK: store float 0.000000e+00, ptr %[[PRIVATE]]

// Call to the reduction function.
// CHECK: call i32 @__kmpc_reduce
// CHECK-SAME: @[[REDFUNC:[A-Za-z_.][A-Za-z0-9_.]*]]

// Atomic reduction.
// CHECK: %[[PARTIAL:.+]] = load float, ptr %[[PRIVATE]]
// CHECK: atomicrmw fadd ptr %{{.*}}, float %[[PARTIAL]]

// Non-atomic reduction:
// CHECK: fadd float
// CHECK: call void @__kmpc_end_reduce
// CHECK: br label %[[FINALIZE:.+]]

// CHECK: [[FINALIZE]]:
// CHECK: call void @__kmpc_barrier

// Update of the private variable using the reduction region
// (the body block currently comes after all the other blocks).
// CHECK: %[[PARTIAL:.+]] = load float, ptr %[[PRIVATE]]
// CHECK: %[[UPDATED:.+]] = fadd float %[[PARTIAL]], 2.000000e+00
// CHECK: store float %[[UPDATED]], ptr %[[PRIVATE]]

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: fadd float
