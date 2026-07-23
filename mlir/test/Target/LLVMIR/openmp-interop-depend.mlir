// RUN: rm -rf %t && split-file %s %t
// RUN: not mlir-translate --mlir-to-llvmir %t/init.mlir 2>&1 | FileCheck %s --check-prefix=INIT
// RUN: not mlir-translate --mlir-to-llvmir %t/use.mlir 2>&1 | FileCheck %s --check-prefix=USE
// RUN: not mlir-translate --mlir-to-llvmir %t/destroy.mlir 2>&1 | FileCheck %s --check-prefix=DESTROY

// The depend clause on interop operations is not yet handled by the
// translation to LLVM IR; each action op must emit a clean diagnostic.

//--- init.mlir
// INIT: error: not yet implemented: Unhandled clause depend in omp.interop.init operation
llvm.func @test_interop_init_depend(%interop: !llvm.ptr, %dep: !llvm.ptr) {
  omp.interop.init %interop : !llvm.ptr interop_types([#omp<interop_type(targetsync)>]) depend(taskdependinout -> %dep : !llvm.ptr)
  llvm.return
}

//--- use.mlir
// USE: error: not yet implemented: Unhandled clause depend in omp.interop.use operation
llvm.func @test_interop_use_depend(%interop: !llvm.ptr, %dep: !llvm.ptr) {
  omp.interop.use %interop : !llvm.ptr depend(taskdependinout -> %dep : !llvm.ptr)
  llvm.return
}

//--- destroy.mlir
// DESTROY: error: not yet implemented: Unhandled clause depend in omp.interop.destroy operation
llvm.func @test_interop_destroy_depend(%interop: !llvm.ptr, %dep: !llvm.ptr) {
  omp.interop.destroy %interop : !llvm.ptr depend(taskdependinout -> %dep : !llvm.ptr)
  llvm.return
}
