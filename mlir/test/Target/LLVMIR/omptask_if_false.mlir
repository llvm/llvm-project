// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @foo_(%arg0: !llvm.ptr {fir.bindc_name = "n"}, %arg1: !llvm.ptr {fir.bindc_name = "r"}) attributes {fir.internal_name = "_QPfoo"} {
  %0 = llvm.mlir.constant(false) : i1
  omp.task if(%0) depend(taskdependin -> %arg0 : !llvm.ptr) {
    %1 = llvm.load %arg0 : !llvm.ptr -> i32
    llvm.store %1, %arg1 : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK: call void @__kmpc_omp_wait_deps
// CHECK-NEXT: call void @__kmpc_omp_task_begin_if0
// CHECK-NEXT: call void @foo_..omp_par
// CHECK-NEXT: call void @__kmpc_omp_task_complete_if0

