// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Only check the overall shape of the code and the presence of relevant
// runtime calls. Actual IR checking is done at the OpenMPIRBuilder level.

omp.declare_reduction @add_f32 : f32
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
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}

// CHECK-LABEL: @simple_reduction
llvm.func @simple_reduction(%lb : i64, %ub : i64, %step : i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  omp.parallel {
    omp.wsloop reduction(@add_f32 %0 -> %prv : !llvm.ptr) {
      omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
        %1 = llvm.mlir.constant(2.0 : f32) : f32
        %2 = llvm.load %prv : !llvm.ptr -> f32
        %3 = llvm.fadd %1, %2 : f32
        llvm.store %3, %prv : f32, !llvm.ptr
        omp.yield
      }
      omp.terminator
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
// CHECK: %[[UPDATED:.+]] = fadd float 2.000000e+00, %[[PARTIAL]]
// CHECK: store float %[[UPDATED]], ptr %[[PRIVATE]]

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: fadd float

// -----

omp.declare_reduction @add_f32 : f32
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
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}

// When the same reduction declaration is used several times, its regions
// are translated several times, which shouldn't lead to value/block
// remapping assertions.
// CHECK-LABEL: @reuse_declaration
llvm.func @reuse_declaration(%lb : i64, %ub : i64, %step : i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  omp.parallel {
    omp.wsloop reduction(@add_f32 %0 -> %prv0, @add_f32 %2 -> %prv1 : !llvm.ptr, !llvm.ptr) {
      omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
        %1 = llvm.mlir.constant(2.0 : f32) : f32
        %3 = llvm.load %prv0 : !llvm.ptr -> f32
        %4 = llvm.fadd %3, %1 : f32
        llvm.store %4, %prv0 : f32, !llvm.ptr
        %5 = llvm.load %prv1 : !llvm.ptr -> f32
        %6 = llvm.fadd %5, %1 : f32
        llvm.store %6, %prv1 : f32, !llvm.ptr
        omp.yield
      }
      omp.terminator
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
// CHECK: %[[PRIVATE1:.+]] = alloca float
// CHECK: %[[PRIVATE2:.+]] = alloca float
// CHECK: store float 0.000000e+00, ptr %[[PRIVATE1]]
// CHECK: store float 0.000000e+00, ptr %[[PRIVATE2]]

// Call to the reduction function.
// CHECK: call i32 @__kmpc_reduce
// CHECK-SAME: @[[REDFUNC:[A-Za-z_.][A-Za-z0-9_.]*]]

// Atomic reduction.
// CHECK: %[[PARTIAL1:.+]] = load float, ptr %[[PRIVATE1]]
// CHECK: atomicrmw fadd ptr %{{.*}}, float %[[PARTIAL1]]
// CHECK: %[[PARTIAL2:.+]] = load float, ptr %[[PRIVATE2]]
// CHECK: atomicrmw fadd ptr %{{.*}}, float %[[PARTIAL2]]

// Non-atomic reduction:
// CHECK: fadd float
// CHECK: fadd float
// CHECK: call void @__kmpc_end_reduce
// CHECK: br label %[[FINALIZE:.+]]

// CHECK: [[FINALIZE]]:
// CHECK: call void @__kmpc_barrier

// Update of the private variable using the reduction region
// (the body block currently comes after all the other blocks).
// CHECK: %[[PARTIAL1:.+]] = load float, ptr %[[PRIVATE1]]
// CHECK: %[[UPDATED1:.+]] = fadd float %[[PARTIAL1]], 2.000000e+00
// CHECK: store float %[[UPDATED1]], ptr %[[PRIVATE1]]
// CHECK: %[[PARTIAL2:.+]] = load float, ptr %[[PRIVATE2]]
// CHECK: %[[UPDATED2:.+]] = fadd float %[[PARTIAL2]], 2.000000e+00
// CHECK: store float %[[UPDATED2]], ptr %[[PRIVATE2]]

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: fadd float
// CHECK: fadd float


// -----

omp.declare_reduction @add_f32 : f32
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
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}

// It's okay not to reference the reduction variable in the body.
// CHECK-LABEL: @missing_omp_reduction
llvm.func @missing_omp_reduction(%lb : i64, %ub : i64, %step : i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  omp.parallel {
    omp.wsloop reduction(@add_f32 %0 -> %prv0, @add_f32 %2 -> %prv1 : !llvm.ptr, !llvm.ptr) {
      omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
        %1 = llvm.mlir.constant(2.0 : f32) : f32
        %3 = llvm.load %prv0 : !llvm.ptr -> f32
        %4 = llvm.fadd %3, %1 : f32
        llvm.store %4, %prv0 : f32, !llvm.ptr
        omp.yield
      }
      omp.terminator
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
// CHECK: %[[PRIVATE1:.+]] = alloca float
// CHECK: %[[PRIVATE2:.+]] = alloca float
// CHECK: store float 0.000000e+00, ptr %[[PRIVATE1]]
// CHECK: store float 0.000000e+00, ptr %[[PRIVATE2]]

// Call to the reduction function.
// CHECK: call i32 @__kmpc_reduce
// CHECK-SAME: @[[REDFUNC:[A-Za-z_.][A-Za-z0-9_.]*]]

// Atomic reduction.
// CHECK: %[[PARTIAL1:.+]] = load float, ptr %[[PRIVATE1]]
// CHECK: atomicrmw fadd ptr %{{.*}}, float %[[PARTIAL1]]
// CHECK: %[[PARTIAL2:.+]] = load float, ptr %[[PRIVATE2]]
// CHECK: atomicrmw fadd ptr %{{.*}}, float %[[PARTIAL2]]

// Non-atomic reduction:
// CHECK: fadd float
// CHECK: fadd float
// CHECK: call void @__kmpc_end_reduce
// CHECK: br label %[[FINALIZE:.+]]

// CHECK: [[FINALIZE]]:
// CHECK: call void @__kmpc_barrier

// Update of the private variable using the reduction region
// (the body block currently comes after all the other blocks).
// CHECK: %[[PARTIAL1:.+]] = load float, ptr %[[PRIVATE1]]
// CHECK: %[[UPDATED1:.+]] = fadd float %[[PARTIAL1]], 2.000000e+00
// CHECK: store float %[[UPDATED1]], ptr %[[PRIVATE1]]
// CHECK-NOT: %{{.*}} = load float, ptr %[[PRIVATE2]]
// CHECK-NOT: %{{.*}} = fadd float %[[PARTIAL2]], 2.000000e+00

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: fadd float
// CHECK: fadd float

// -----

omp.declare_reduction @add_f32 : f32
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
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}

// It's okay to refer to the same reduction variable more than once in the
// body.
// CHECK-LABEL: @double_reference
llvm.func @double_reference(%lb : i64, %ub : i64, %step : i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  omp.parallel {
    omp.wsloop reduction(@add_f32 %0 -> %prv : !llvm.ptr) {
      omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
        %1 = llvm.mlir.constant(2.0 : f32) : f32
        %2 = llvm.load %prv : !llvm.ptr -> f32
        %3 = llvm.fadd %2, %1 : f32
        llvm.store %3, %prv : f32, !llvm.ptr
        %4 = llvm.load %prv : !llvm.ptr -> f32
        %5 = llvm.fadd %4, %1 : f32
        llvm.store %5, %prv : f32, !llvm.ptr
        omp.yield
      }
      omp.terminator
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
// CHECK: %[[PARTIAL:.+]] = load float, ptr %[[PRIVATE]]
// CHECK: %[[UPDATED:.+]] = fadd float %[[PARTIAL]], 2.000000e+00
// CHECK: store float %[[UPDATED]], ptr %[[PRIVATE]]

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: fadd float

// -----

omp.declare_reduction @add_f32 : f32
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
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}

omp.declare_reduction @mul_f32 : f32
init {
^bb0(%arg: f32):
  %0 = llvm.mlir.constant(1.0 : f32) : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = llvm.fmul %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}

// CHECK-LABEL: @no_atomic
llvm.func @no_atomic(%lb : i64, %ub : i64, %step : i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  omp.parallel {
    omp.wsloop reduction(@add_f32 %0 -> %prv0, @mul_f32 %2 -> %prv1 : !llvm.ptr, !llvm.ptr) {
      omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
        %1 = llvm.mlir.constant(2.0 : f32) : f32
        %3 = llvm.load %prv0 : !llvm.ptr -> f32
        %4 = llvm.fadd %3, %1 : f32
        llvm.store %4, %prv0 : f32, !llvm.ptr
        %5 = llvm.load %prv1 : !llvm.ptr -> f32
        %6 = llvm.fmul %5, %1 : f32
        llvm.store %6, %prv1 : f32, !llvm.ptr
        omp.yield
      }
      omp.terminator
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
// CHECK: %[[PRIVATE1:.+]] = alloca float
// CHECK: %[[PRIVATE2:.+]] = alloca float
// CHECK: store float 0.000000e+00, ptr %[[PRIVATE1]]
// CHECK: store float 1.000000e+00, ptr %[[PRIVATE2]]

// Call to the reduction function.
// CHECK: call i32 @__kmpc_reduce
// CHECK-SAME: @[[REDFUNC:[A-Za-z_.][A-Za-z0-9_.]*]]

// Atomic reduction not provided.
// CHECK: unreachable

// Non-atomic reduction:
// CHECK: fadd float
// CHECK: fmul float
// CHECK: call void @__kmpc_end_reduce
// CHECK: br label %[[FINALIZE:.+]]

// CHECK: [[FINALIZE]]:
// CHECK: call void @__kmpc_barrier

// Update of the private variable using the reduction region
// (the body block currently comes after all the other blocks).
// CHECK: %[[PARTIAL1:.+]] = load float, ptr %[[PRIVATE1]]
// CHECK: %[[UPDATED1:.+]] = fadd float %[[PARTIAL1]], 2.000000e+00
// CHECK: store float %[[UPDATED1]], ptr %[[PRIVATE1]]
// CHECK: %[[PARTIAL2:.+]] = load float, ptr %[[PRIVATE2]]
// CHECK: %[[UPDATED2:.+]] = fmul float %[[PARTIAL2]], 2.000000e+00
// CHECK: store float %[[UPDATED2]], ptr %[[PRIVATE2]]

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: fadd float
// CHECK: fmul float

// -----

omp.declare_reduction @add_f32 : f32
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
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}

// CHECK-LABEL: @simple_reduction_parallel
llvm.func @simple_reduction_parallel() {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  omp.parallel reduction(@add_f32 %0 -> %prv : !llvm.ptr) {
    %1 = llvm.mlir.constant(2.0 : f32) : f32
    %2 = llvm.load %prv : !llvm.ptr -> f32
    %3 = llvm.fadd %2, %1 : f32
    llvm.store %3, %prv : f32, !llvm.ptr
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

// Update of the private variable
// CHECK: %[[PARTIAL:.+]] = load float, ptr %[[PRIVATE]]
// CHECK: %[[UPDATED:.+]] = fadd float %[[PARTIAL]], 2.000000e+00
// CHECK: store float %[[UPDATED]], ptr %[[PRIVATE]]

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

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: fadd float

// -----

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg: i32):
  %0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield (%0 : i32)
}
combiner {
^bb1(%arg0: i32, %arg1: i32):
  %1 = llvm.add %arg0, %arg1 : i32
  omp.yield (%1 : i32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> i32
  llvm.atomicrmw add %arg2, %2 monotonic : !llvm.ptr, i32
  omp.yield
}

// CHECK-LABEL: @parallel_nested_workshare_reduction
llvm.func @parallel_nested_workshare_reduction(%ub : i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr

  %lb = llvm.mlir.constant(1 : i64) : i64
  %step = llvm.mlir.constant(1 : i64) : i64
  
  omp.parallel {
    omp.wsloop reduction(@add_i32 %0 -> %prv : !llvm.ptr) {
      omp.loop_nest (%iv) : i64 = (%lb) to (%ub) step (%step) {
        %ival = llvm.trunc %iv : i64 to i32
        %lprv = llvm.load %prv : !llvm.ptr -> i32
        %add = llvm.add %lprv, %ival : i32
        llvm.store %add, %prv : i32, !llvm.ptr
        omp.yield
      }
      omp.terminator
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
// CHECK: %[[PRIVATE:[0-9]+]] = alloca i32
// CHECK: store i32 0, ptr %[[PRIVATE]]

// Loop exit:
// CHECK: call void @__kmpc_barrier

// Call to the reduction function.
// CHECK: call i32 @__kmpc_reduce
// CHECK-SAME: @[[REDFUNC:[A-Za-z_.][A-Za-z0-9_.]*]]

// Atomic reduction:
// CHECK: %[[PARTIAL:.+]] = load i32, ptr %[[PRIVATE]]
// CHECK: atomicrmw add ptr %{{.*}}, i32 %[[PARTIAL]]

// Non-atomic reduction:
// CHECK: add i32
// CHECK: call void @__kmpc_end_reduce

// Update of the private variable using the reduction region
// (the body block currently comes after all the other blocks).
// CHECK: %[[PARTIAL:.+]] = load i32, ptr %[[PRIVATE]]
// CHECK: %[[UPDATED:.+]] = add i32 %[[PARTIAL]], {{.*}}
// CHECK: store i32 %[[UPDATED]], ptr %[[PRIVATE]]

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: add i32
