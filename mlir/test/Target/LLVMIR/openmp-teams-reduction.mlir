// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Only check the overall shape of the code and the presence of relevant
// runtime calls. Actual IR checking is done at the OpenMPIRBuilder level.

omp.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%0 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = llvm.add %arg0, %arg1 : i32
  omp.yield(%0 : i32)
}
llvm.func @simple_teams_only_reduction_() attributes {fir.internal_name = "_QPsimple_teams_only_reduction", frame_pointer = #llvm.framePointerKind<all>, target_cpu = "x86-64"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "sum"} : (i64) -> !llvm.ptr
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x i32 {bindc_name = "index_"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(0 : index) : i64
  %5 = llvm.mlir.constant(10000 : index) : i64
  %6 = llvm.mlir.constant(1 : index) : i64
  %7 = llvm.mlir.constant(0 : i32) : i32
  %8 = llvm.mlir.constant(1 : i64) : i64
  %9 = llvm.mlir.constant(1 : i64) : i64
  llvm.store %7, %1 : i32, !llvm.ptr
  omp.teams reduction(@add_reduction_i32 %1 -> %arg0 : !llvm.ptr) {
    %10 = llvm.trunc %6 : i64 to i32
    llvm.br ^bb1(%10, %5 : i32, i64)
  ^bb1(%11: i32, %12: i64):  // 2 preds: ^bb0, ^bb2
    %13 = llvm.icmp "sgt" %12, %4 : i64
    llvm.cond_br %13, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.store %11, %3 : i32, !llvm.ptr
    %14 = llvm.load %arg0 : !llvm.ptr -> i32
    %15 = llvm.load %3 : !llvm.ptr -> i32
    %16 = llvm.add %14, %15 : i32
    llvm.store %16, %arg0 : i32, !llvm.ptr
    %17 = llvm.load %3 : !llvm.ptr -> i32
    %18 = llvm.add %17, %10 overflow<nsw> : i32
    %19 = llvm.sub %12, %6 : i64
    llvm.br ^bb1(%18, %19 : i32, i64)
  ^bb3:  // pred: ^bb1
    llvm.store %11, %3 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// Allocate reduction array
// CHECK: %[[REDARRAY:[A-Za-z_.][A-Za-z0-9_.]*]] = alloca [1 x ptr], align 8
// Call to outlined function
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_teams
// CHECK-SAME: @[[OUTLINED:[A-Za-z_.][A-Za-z0-9_.]*]]
// Outlined function.

// Private reduction variable and its initialization.

// Call to the reduction function.
// CHECK: call i32 @__kmpc_reduce
// Check that the reduction array is passed in.
// CHECK-SAME: %[[REDARRAY]]
// CHECK-SAME: @[[REDFUNC:[A-Za-z_.][A-Za-z0-9_.]*]]

// CHECK: [[FINALIZE:.+]]:
// CHECK: call void @__kmpc_barrier

// Non atomic version
// CHECK: call void @__kmpc_end_reduce
// CHECK: br label %[[FINALIZE]]

// Atomic version not generated
// CHECK: unreachable

// CHECK: define internal void @[[OUTLINED]]

// Reduction function.
// CHECK: define internal void @[[REDFUNC]]
// CHECK: add i32
