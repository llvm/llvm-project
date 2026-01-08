// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Tests that no privatization barrier is added for operations inside of SINGLE.
// The specific combination here (wsloop inside of single) may not be valid
// OpenMP. This combination was chosen so that this patch did not have to wait
// for taskloop support. If it ever becomes a problem having a wsloop inside
// of single, this can be updated to use taskloop.

omp.private {type = private} @_QFwsloop_privateEi_private_ref_i32 : i32

llvm.func @foo_free(!llvm.ptr)

omp.private {type = firstprivate} @_QFwsloop_privateEc_firstprivate_ref_c8 : !llvm.array<1 x i8> copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.array<1 x i8>
  llvm.store %0, %arg1 : !llvm.array<1 x i8>, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  llvm.call @foo_free(%arg0) : (!llvm.ptr) -> ()
  omp.yield
}

llvm.func @wsloop_private_(%arg0: !llvm.ptr {fir.bindc_name = "y"}) attributes {fir.internal_name = "_QPwsloop_private", frame_pointer = #llvm.framePointerKind<all>, target_cpu = "x86-64"} {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x f32 {bindc_name = "x"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %5 = llvm.alloca %0 x !llvm.array<1 x i8> {bindc_name = "c"} : (i64) -> !llvm.ptr
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.mlir.constant(10 : i32) : i32
  %8 = llvm.mlir.constant(0 : i32) : i32
  omp.parallel {
    omp.single {
      omp.wsloop private(@_QFwsloop_privateEc_firstprivate_ref_c8 %5 -> %arg1, @_QFwsloop_privateEi_private_ref_i32 %3 -> %arg2 : !llvm.ptr, !llvm.ptr) private_barrier {
  // CHECK: omp.private.copy:
  // CHECK-NOT: __kmpc_barrier
  // CHECK: br label
        omp.loop_nest (%arg4) : i32 = (%8) to (%7) inclusive step (%6) {
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
