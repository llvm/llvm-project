// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verify that an inscan reduction on a worksharing loop nested inside two
// parallel regions allocates its shared temporary buffer inside the *outer*
// parallel's outlined region, not in the enclosing function. If the buffer
// (pointer) were allocated in the top-level function it would be shared by
// every nested inner-parallel team, so independent teams would race on and
// double-free the same buffer.

omp.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%0 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = llvm.add %arg0, %arg1 : i32
  omp.yield(%0 : i32)
}
llvm.func @nested_scan_reduction() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %5 = llvm.alloca %0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
  %10 = llvm.mlir.constant(100 : i32) : i32
  %11 = llvm.mlir.constant(1 : i32) : i32
  %14 = llvm.mlir.addressof @_QFEa : !llvm.ptr
  %15 = llvm.mlir.addressof @_QFEb : !llvm.ptr
  omp.parallel {
    omp.parallel {
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

// The enclosing function only forks the outer parallel; the scan buffer pointer
// must NOT be allocated here (that would be shared across all nested teams).
// CHECK-LABEL: define void @nested_scan_reduction()
// CHECK-NOT: alloca ptr
// CHECK: call void {{.*}}@__kmpc_fork_call({{.*}} @nested_scan_reduction..omp_par.1

// The buffer pointer is allocated inside the outer parallel's outlined region,
// so each outer-parallel thread has its own buffer for its nested inner team.
// CHECK: define internal void @nested_scan_reduction..omp_par.1
// CHECK: %vla = alloca ptr
// CHECK: call void {{.*}}@__kmpc_fork_call({{.*}} @nested_scan_reduction..omp_par,
