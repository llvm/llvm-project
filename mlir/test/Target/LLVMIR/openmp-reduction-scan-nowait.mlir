// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verify buffer-lifetime handling for an inclusive scan reduction on a
// worksharing loop with `nowait`. The scan (second) loop reads each thread's
// prefix value from the shared scan buffer, and a masked region then frees
// that buffer. A barrier must be emitted *before* the masked region so the
// buffer cannot be freed while another thread is still reading it (a masked
// region is not a barrier). Because of `nowait`, no end-of-construct barrier
// is emitted *after* the masked region.

omp.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%0 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = llvm.add %arg0, %arg1 : i32
  omp.yield(%0 : i32)
}
llvm.func @scan_reduction_nowait() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %5 = llvm.alloca %0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
  %10 = llvm.mlir.constant(100 : i32) : i32
  %11 = llvm.mlir.constant(1 : i32) : i32
  %14 = llvm.mlir.addressof @_QFEa : !llvm.ptr
  %15 = llvm.mlir.addressof @_QFEb : !llvm.ptr
  omp.parallel {
    %37 = llvm.mlir.constant(1 : i64) : i64
    %38 = llvm.alloca %37 x i32 {bindc_name = "k", pinned} : (i64) -> !llvm.ptr
    omp.wsloop nowait reduction(mod: inscan, @add_reduction_i32 %5 -> %arg0 : !llvm.ptr) {
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

// CHECK-LABEL: define internal void @scan_reduction_nowait..omp_par
// After the scan (second) loop, the mandatory buffer-lifetime barrier is
// emitted before the masked region that frees the shared scan buffer.
// CHECK: omp.scan.loop.cont:
// CHECK: call void @__kmpc_barrier
// CHECK: call i32 @__kmpc_masked
// CHECK: call void @free(ptr
// CHECK: call void @__kmpc_end_masked
// With `nowait`, there is no end-of-construct barrier after the buffer is freed.
// CHECK-NOT: call void @__kmpc_barrier
// CHECK: ret void
