// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Basic scope: body runs in omp.scope.region, barrier emitted after.
// CHECK-LABEL: define internal void @scope_basic..omp_par
// CHECK:         br label %omp.scope.region
// CHECK:       omp.scope.region:
// CHECK:         br label %omp.region.cont3
// CHECK:       omp_region.finalize:
// CHECK:         call void @__kmpc_barrier(

llvm.func @scope_basic() {
  omp.parallel {
    omp.scope {
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// Scope nowait: body runs in omp.scope.region, no barrier emitted.
// CHECK-LABEL: define internal void @scope_nowait..omp_par
// CHECK:         br label %omp.scope.region
// CHECK:       omp.scope.region:
// CHECK:       omp_region.finalize:
// CHECK-NOT:     call void @__kmpc_barrier(
// CHECK:         ret void

llvm.func @scope_nowait() {
  omp.parallel {
    omp.scope nowait {
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// Scope with reduction: reduction vars initialized before scope body,
// __kmpc_reduce / __kmpc_end_reduce emitted + scope barrier
// CHECK-LABEL: define internal void @scope_reduction..omp_par
// CHECK:       omp.reduction.init:
// CHECK:         store float 0.000000e+00, ptr
// CHECK:       omp.scope.region:
// CHECK:         fadd float
// CHECK:       omp_region.finalize:
// CHECK:         call void @__kmpc_barrier(
// CHECK:         call i32 @__kmpc_reduce(
// CHECK:       reduce.switch.nonatomic:
// CHECK:         fadd float
// CHECK:         call void @__kmpc_end_reduce(

omp.declare_reduction @add_f32 : f32 init {
^bb0(%arg0: f32):
  %c = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield(%c : f32)
} combiner {
^bb0(%arg0: f32, %arg1: f32):
  %r = llvm.fadd %arg0, %arg1 : f32
  omp.yield(%r : f32)
}

llvm.func @scope_reduction(%ptr: !llvm.ptr) {
  omp.parallel {
    omp.scope reduction(@add_f32 %ptr -> %arg0 : !llvm.ptr) {
      %c = llvm.mlir.constant(1.0 : f32) : f32
      %v = llvm.load %arg0 : !llvm.ptr -> f32
      %r = llvm.fadd %v, %c : f32
      llvm.store %r, %arg0 : f32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// Scope with reduction + nowait: nowait suppresses the scope barrier
// __kmpc_reduce_nowait / __kmpc_end_reduce_nowait emitted
// CHECK-LABEL: define internal void @scope_reduction_nowait..omp_par
// CHECK:       omp.reduction.init:
// CHECK:         store float 0.000000e+00, ptr
// CHECK:       omp.scope.region:
// CHECK:         fadd float
// CHECK:       omp_region.finalize:
// CHECK-NOT:     call void @__kmpc_barrier(
// CHECK:         call i32 @__kmpc_reduce_nowait(
// CHECK:       reduce.switch.nonatomic:
// CHECK:         call void @__kmpc_end_reduce_nowait(

omp.declare_reduction @add_f32_2 : f32 init {
^bb0(%arg0: f32):
  %c = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield(%c : f32)
} combiner {
^bb0(%arg0: f32, %arg1: f32):
  %r = llvm.fadd %arg0, %arg1 : f32
  omp.yield(%r : f32)
}

llvm.func @scope_reduction_nowait(%ptr: !llvm.ptr) {
  omp.parallel {
    omp.scope nowait reduction(@add_f32_2 %ptr -> %arg0 : !llvm.ptr) {
      %c = llvm.mlir.constant(1.0 : f32) : f32
      %v = llvm.load %arg0 : !llvm.ptr -> f32
      %r = llvm.fadd %v, %c : f32
      llvm.store %r, %arg0 : f32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// Scope with private: a per-thread alloca is created, body uses it,
// original variable is not touched, barrier emitted after.
// CHECK-LABEL: define internal void @scope_private..omp_par
// CHECK:         %omp.private.alloc = alloca i32
// CHECK:       omp.private.init:
// CHECK:       omp.scope.region:
// CHECK:         store i32 1, ptr %omp.private.alloc
// CHECK:       omp_region.finalize:
// CHECK:         call void @__kmpc_barrier(

omp.private {type = private} @x_private_i32 : i32

llvm.func @scope_private(%x_ptr: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.scope private(@x_private_i32 %x_ptr -> %arg0 : !llvm.ptr) {
      llvm.store %c1, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// Scope with firstprivate: per-thread alloca created, original value copied in
// before the body, barrier emitted after.
// CHECK-LABEL: define internal void @scope_firstprivate..omp_par
// CHECK:         %omp.private.alloc = alloca i32
// CHECK:       omp.private.copy:
// CHECK:         %[[ORIG:.*]] = load i32, ptr
// CHECK:         store i32 %[[ORIG]], ptr %omp.private.alloc
// CHECK:       omp.scope.region:
// CHECK:         load i32, ptr %omp.private.alloc
// CHECK:       omp_region.finalize:
// CHECK:         call void @__kmpc_barrier(

omp.private {type = firstprivate} @x_firstprivate_i32 : i32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> i32
  llvm.store %0, %arg1 : i32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
}

llvm.func @scope_firstprivate(%x_ptr: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.scope private(@x_firstprivate_i32 %x_ptr -> %arg0 : !llvm.ptr) {
      %v = llvm.load %arg0 : !llvm.ptr -> i32
      %r = llvm.add %v, %c1 : i32
      llvm.store %r, %arg0 : i32, !llvm.ptr
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
