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

// CHECK-LABEL wsloop_private_
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


// Check for a case where  the barrier should be applied. Regression check for
// infinite loop.
omp.private {type = private} @_QFsubEi_private_i32 : i32
omp.private {type = firstprivate} @_QFsubEp_firstprivate_i32 : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}

// CHECK-LABEL: _QPsub
llvm.func @_QPsub() {
  %0 = llvm.mlir.constant(100 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x i32 {bindc_name = "p"} : (i64) -> !llvm.ptr
  %4 = llvm.alloca %2 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  llvm.store %0, %3 : i32, !llvm.ptr
  omp.wsloop private(@_QFsubEp_firstprivate_i32 %3 -> %arg0, @_QFsubEi_private_i32 %4 -> %arg1 : !llvm.ptr, !llvm.ptr) private_barrier {
  // CHECK: omp.private.copy:
  // CHECK: __kmpc_barrier
  // CHECK: br label
    omp.loop_nest (%arg2) : i32 = (%1) to (%1) inclusive step (%1) {
      llvm.store %arg2, %arg1 : i32, !llvm.ptr
      %5 = llvm.add %arg2, %1 : i32
      %6 = llvm.icmp "sgt" %5, %1 : i32
      llvm.cond_br %6, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      llvm.store %5, %arg1 : i32, !llvm.ptr
      %7 = llvm.load %arg0 : !llvm.ptr -> i32
      llvm.store %7, %3 : i32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      omp.yield
    }
  }
  llvm.return
}
