// RUN: not mlir-translate --mlir-to-llvmir %s 2>&1 | FileCheck %s

// CHECK: error: not yet implemented: Unhandled clause depend in omp.interop.init operation
llvm.func @test_interop_init_depend(%interop: !llvm.ptr, %dep: !llvm.ptr) {
  omp.interop.init %interop : !llvm.ptr interop_types([#omp<interop_type(targetsync)>]) depend(taskdependinout -> %dep : !llvm.ptr)
  llvm.return
}
