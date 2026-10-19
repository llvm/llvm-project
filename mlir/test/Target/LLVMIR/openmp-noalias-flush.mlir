// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Accesses through `noalias` pointer arguments (Fortran dummy arguments) must
// not be moved across points where OpenMP requires a flush. At those points the
// `noalias` arguments are passed to an empty inline asm.

// CHECK-LABEL: define void @barrier(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], ptr %{{[0-9]+}})
// CHECK: call void asm sideeffect "", "r,~{memory}"(ptr %[[A]])
// CHECK: call void @__kmpc_barrier
llvm.func @barrier(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr) {
  omp.barrier
  llvm.return
}

// CHECK-LABEL: define void @critical(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]])
// CHECK: call void asm sideeffect "", "r,~{memory}"(ptr %[[A]])
// CHECK: call void @__kmpc_critical
llvm.func @critical(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.critical {
    omp.terminator
  }
  llvm.return
}

// The flush of single comes from its end barrier.
// CHECK-LABEL: define void @single(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]])
// CHECK: call void asm sideeffect "", "r,~{memory}"(ptr %[[A]])
// CHECK: call i32 @__kmpc_single
llvm.func @single(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.single {
    omp.terminator
  }
  llvm.return
}

// With nowait there is no end barrier, so nothing is pinned.
// CHECK-LABEL: define void @single_nowait(ptr noalias %{{[0-9]+}})
// CHECK-NOT: call void asm sideeffect
// CHECK: ret void
llvm.func @single_nowait(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.single nowait {
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @wsloop(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}})
// CHECK: call void asm sideeffect "", "r,~{memory}"(ptr %[[A]])
// CHECK: call void @__kmpc_for_static_init
// CHECK: call void @__kmpc_barrier
llvm.func @wsloop(%arg0: !llvm.ptr {llvm.noalias}, %lb: i32, %ub: i32, %step: i32) {
  omp.wsloop {
    omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
      omp.yield
    }
  }
  llvm.return
}

// A flush without a list pins all `noalias` pointer arguments.
// CHECK-LABEL: define void @flush(
// CHECK-SAME: ptr noalias %[[A:[0-9]+]], ptr noalias %[[B:[0-9]+]], ptr %{{[0-9]+}})
// CHECK: call void asm sideeffect "", "r,r,~{memory}"(ptr %[[A]], ptr %[[B]])
// CHECK: call void @__kmpc_flush
llvm.func @flush(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}, %arg2: !llvm.ptr) {
  omp.flush
  llvm.return
}

// A flush with a list pins only the listed `noalias` arguments, once each.
// CHECK-LABEL: define void @flush_list(
// CHECK-SAME: ptr noalias %{{[0-9]+}}, ptr noalias %[[B:[0-9]+]], ptr %{{[0-9]+}})
// CHECK: call void asm sideeffect "", "r,~{memory}"(ptr %[[B]])
// CHECK: call void @__kmpc_flush
llvm.func @flush_list(%arg0: !llvm.ptr {llvm.noalias}, %arg1: !llvm.ptr {llvm.noalias}, %arg2: !llvm.ptr) {
  omp.flush (%arg1, %arg2, %arg1 : !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.return
}

// Constructs in a region that is outlined into another function do not pin
// the arguments of the enclosing function.
// CHECK-LABEL: define void @in_parallel(ptr noalias %{{[0-9]+}})
// CHECK-NOT: call void asm sideeffect
// CHECK: define internal void @in_parallel..omp_par
// CHECK-NOT: call void asm sideeffect
// CHECK: call void @__kmpc_barrier
// CHECK-NOT: call void asm sideeffect
// CHECK: call void @__kmpc_flush
// CHECK-NOT: call void asm sideeffect
llvm.func @in_parallel(%arg0: !llvm.ptr {llvm.noalias}) {
  omp.parallel {
    omp.barrier
    omp.flush
    omp.terminator
  }
  llvm.return
}
