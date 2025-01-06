// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// tests a wsloop private + firstprivate + reduction to make sure block structure
// is handled properly.

omp.private {type = private} @_QFwsloop_privateEi_private_ref_i32 : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x i32 {bindc_name = "i", pinned} : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
}

llvm.func @foo_free(!llvm.ptr)

omp.private {type = firstprivate} @_QFwsloop_privateEc_firstprivate_ref_c8 : !llvm.ptr alloc {
^bb0(%arg0: !llvm.ptr):
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x !llvm.array<1 x i8> {bindc_name = "c", pinned} : (i64) -> !llvm.ptr
  omp.yield(%1 : !llvm.ptr)
} copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.array<1 x i8>
  llvm.store %0, %arg1 : !llvm.array<1 x i8>, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
} dealloc {
^bb0(%arg0: !llvm.ptr):
  llvm.call @foo_free(%arg0) : (!llvm.ptr) -> ()
  omp.yield
}

omp.declare_reduction @max_f32 : f32 init {
^bb0(%arg0: f32):
  %0 = llvm.mlir.constant(-3.40282347E+38 : f32) : f32
  omp.yield(%0 : f32)
} combiner {
^bb0(%arg0: f32, %arg1: f32):
  %0 = llvm.intr.maxnum(%arg0, %arg1) {fastmathFlags = #llvm.fastmath<contract>} : (f32, f32) -> f32
  omp.yield(%0 : f32)
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
    omp.wsloop private(@_QFwsloop_privateEc_firstprivate_ref_c8 %5 -> %arg1, @_QFwsloop_privateEi_private_ref_i32 %3 -> %arg2 : !llvm.ptr, !llvm.ptr) reduction(@max_f32 %1 -> %arg3 : !llvm.ptr) {
      omp.loop_nest (%arg4) : i32 = (%8) to (%7) inclusive step (%6) {
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK: call void {{.*}} @__kmpc_fork_call(ptr @1, i32 1, ptr @[[OUTLINED:.*]], ptr %{{.*}})

// CHECK: define internal void @[[OUTLINED:.*]]{{.*}} {

// First, check that all memory for privates and reductions is allocated.
// CHECK: omp.par.entry:
// CHECK:   %[[CHR:.*]] = alloca [1 x i8], i64 1, align 1
// CHECK:   %[[INT:.*]] = alloca i32, i64 1, align 4
// CHECK:   %[[FLT:.*]] = alloca float, align 4
// CHECK:   %[[RED_ARR:.*]] = alloca [1 x ptr], align 8
// CHECK:   br label %[[LATE_ALLOC_BB:.*]]

// CHECK: [[LATE_ALLOC_BB]]:
// CHECK:   br label %[[PRIVATE_CPY_BB:.*]]

// Second, check that first private was properly copied.
// CHECK: [[PRIVATE_CPY_BB:.*]]:
// CHECK:   %[[CHR_VAL:.*]] = load [1 x i8], ptr %{{.*}}, align 1
// CHECK:   store [1 x i8] %[[CHR_VAL]], ptr %[[CHR]], align 1
// CHECK:   br label %[[RED_INIT_BB:.*]]

// Third, check that reduction init took place.
// CHECK: [[RED_INIT_BB]]:
// CHECK:   store float 0x{{.*}}, ptr %[[FLT]], align 4

// Finally, check for the private dealloc region
// CHECK:   call void @foo_free(ptr %[[CHR]])

// CHECK: }
