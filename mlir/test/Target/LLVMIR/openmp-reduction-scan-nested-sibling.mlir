// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verify that an inscan reduction on a worksharing loop bound to an *outer*
// parallel allocates its shared temporary buffer in the enclosing function
// (shared across the outer team through the fork's struct argument), even when
// a sibling inner parallel precedes the loop. The sibling parallel marks the
// outer parallel's alloca frame; that mark must not divert the scan buffer into
// the outer parallel's per-thread scope, which would make the single masked
// thread's allocation invisible to the other team threads and crash at runtime.

omp.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%0 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = llvm.add %arg0, %arg1 : i32
  omp.yield(%0 : i32)
}
llvm.func @sibling_scan_reduction() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %5 = llvm.alloca %0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
  %6 = llvm.alloca %0 x i32 {bindc_name = "v"} : (i64) -> !llvm.ptr
  %10 = llvm.mlir.constant(100 : i32) : i32
  %11 = llvm.mlir.constant(1 : i32) : i32
  %14 = llvm.mlir.addressof @_QFEa : !llvm.ptr
  %15 = llvm.mlir.addressof @_QFEb : !llvm.ptr
  omp.parallel {
    // Sibling inner parallel that precedes the scan loop.
    omp.parallel {
      llvm.store %11, %6 : i32, !llvm.ptr
      omp.terminator
    }
    %38 = llvm.alloca %0 x i32 {bindc_name = "k", pinned} : (i64) -> !llvm.ptr
    omp.wsloop reduction(mod: inscan, @add_reduction_i32 %5 -> %arg0 : !llvm.ptr) {
      omp.loop_nest (%arg1) : i32 = (%11) to (%10) inclusive step (%11) {
        llvm.store %arg1, %38 : i32, !llvm.ptr
        %40 = llvm.load %arg0 : !llvm.ptr -> i32
        %41 = llvm.load %38 : !llvm.ptr -> i32
        %42 = llvm.sext %41 : i32 to i64
        %50 = llvm.getelementptr %14[%42] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %51 = llvm.load %50 : !llvm.ptr -> i32
        %52 = llvm.add %40, %51 : i32
        llvm.store %52, %arg0 : i32, !llvm.ptr
        omp.scan inclusive(%arg0 : !llvm.ptr)
        llvm.store %arg1, %38 : i32, !llvm.ptr
        %53 = llvm.load %arg0 : !llvm.ptr -> i32
        %54 = llvm.load %38 : !llvm.ptr -> i32
        %55 = llvm.sext %54 : i32 to i64
        %63 = llvm.getelementptr %15[%55] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        llvm.store %53, %63 : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}
llvm.mlir.global internal @_QFEa() {addr_space = 0 : i32} : !llvm.array<100 x i32> {
  %0 = llvm.mlir.zero : !llvm.array<100 x i32>
  llvm.return %0 : !llvm.array<100 x i32>
}
llvm.mlir.global internal @_QFEb() {addr_space = 0 : i32} : !llvm.array<100 x i32> {
  %0 = llvm.mlir.zero : !llvm.array<100 x i32>
  llvm.return %0 : !llvm.array<100 x i32>
}

// The scan buffer pointer is allocated in the enclosing function and shared
// across the outer team through the fork_call struct argument.
// CHECK-LABEL: define void @sibling_scan_reduction()
// CHECK: %vla = alloca ptr
// CHECK: call void {{.*}}@__kmpc_fork_call({{.*}} @sibling_scan_reduction..omp_par.1

// The outer parallel's outlined region must NOT allocate the buffer pointer; it
// loads the shared pointer from the fork's struct argument instead.
// CHECK: define internal void @sibling_scan_reduction..omp_par.1
// CHECK-NOT: = alloca ptr
