// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Only check the overall shape of the code and the presence of relevant
// runtime calls. Actual IR checking is done at the OpenMPIRBuilder level.

omp.private {type = private} @_QFsimple_teams_reductionEindex__private_i32 : i32
omp.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%0 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = llvm.add %arg0, %arg1 : i32
  omp.yield(%0 : i32)
}
llvm.func @simple_teams_reduction_() attributes {fir.internal_name = "_QPsimple_teams_reduction", frame_pointer = #llvm.framePointerKind<all>, target_cpu = "x86-64"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "sum"} : (i64) -> !llvm.ptr
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x i32 {bindc_name = "index_"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(10000 : i32) : i32
  %5 = llvm.mlir.constant(1 : i32) : i32
  %6 = llvm.mlir.constant(0 : i32) : i32
  %7 = llvm.mlir.constant(1 : i64) : i64
  %8 = llvm.mlir.constant(1 : i64) : i64
  llvm.store %6, %1 : i32, !llvm.ptr
  omp.teams reduction(@add_reduction_i32 %1 -> %arg0 : !llvm.ptr) {
    omp.distribute private(@_QFsimple_teams_reductionEindex__private_i32 %3 -> %arg1 : !llvm.ptr) {
      omp.loop_nest (%arg2) : i32 = (%5) to (%4) inclusive step (%5) {
        llvm.store %arg2, %arg1 : i32, !llvm.ptr
        %9 = llvm.load %arg0 : !llvm.ptr -> i32
        %10 = llvm.load %arg1 : !llvm.ptr -> i32
        %11 = llvm.add %9, %10 : i32
        llvm.store %11, %arg0 : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}
// Call to outlined function
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_teams
// CHECK-SAME: @[[OUTLINED:[A-Za-z_.][A-Za-z0-9_.]*]]

// Outlined function.
// CHECK: define internal void @[[OUTLINED]]

// Private reduction variable and its initialization.
// CHECK: %[[PRIVATE:.+]] = alloca i32
// CHECK: store i32 0, ptr %[[PRIVATE]]

// Call to the reduction function.
// CHECK: call i32 @__kmpc_reduce
// CHECK-SAME: @[[REDFUNC:[A-Za-z_.][A-Za-z0-9_.]*]]

// Atomic version not generated
// CHECK: unreachable

// Non atomic version
// CHECK: call void @__kmpc_end_reduce

// Finalize
// CHECK: br label %[[FINALIZE:.+]]

// CHECK: [[FINALIZE]]:
// CHECK: call void @__kmpc_barrier

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: add i32
